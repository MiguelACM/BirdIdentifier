// Minimal EXIF reader: finds the GPS position and the date a JPEG photo was taken.
// Runs entirely in the browser; nothing is sent anywhere. Returns
// { lat, lon, month } with null for anything missing (most photos shared through
// messaging apps or social media have their EXIF data stripped).

const TAG_EXIF_IFD = 0x8769;
const TAG_GPS_IFD = 0x8825;
const TAG_DATE_ORIGINAL = 0x9003; // "YYYY:MM:DD HH:MM:SS"
const TAG_DATE = 0x0132;
const GPS_LAT_REF = 1, GPS_LAT = 2, GPS_LON_REF = 3, GPS_LON = 4;

export async function readExif(file) {
  const empty = { lat: null, lon: null, month: null };
  try {
    // EXIF lives in the first APP1 segment, near the start of the file
    const view = new DataView(await file.slice(0, 256 * 1024).arrayBuffer());
    const tiff = findTiffStart(view);
    return tiff === null ? empty : parseTiff(view, tiff);
  } catch {
    return empty; // truncated or unusual files: just skip location hints
  }
}

function findTiffStart(view) {
  if (view.getUint16(0) !== 0xffd8) return null; // not a JPEG
  let offset = 2;
  while (offset + 4 <= view.byteLength) {
    const marker = view.getUint16(offset);
    const length = view.getUint16(offset + 2);
    // APP1 starting with "Exif\0\0"
    if (marker === 0xffe1 && view.getUint32(offset + 4) === 0x45786966 && view.getUint16(offset + 8) === 0) {
      return offset + 10;
    }
    if ((marker & 0xff00) !== 0xff00 || marker === 0xffda) return null; // image data started
    offset += 2 + length;
  }
  return null;
}

function parseTiff(view, start) {
  const little = view.getUint16(start) === 0x4949; // "II" = little-endian, "MM" = big-endian
  const u16 = (o) => view.getUint16(start + o, little);
  const u32 = (o) => view.getUint32(start + o, little);

  // Read one IFD (a list of 12-byte tag entries) into {tag: entry offset}
  const readIfd = (ifdOffset) => {
    const entries = {};
    const count = u16(ifdOffset);
    for (let i = 0; i < count; i++) {
      const entry = ifdOffset + 2 + i * 12;
      entries[u16(entry)] = entry;
    }
    return entries;
  };
  // Value offset: values longer than 4 bytes are stored elsewhere
  const valueAt = (entry, size) => (size > 4 ? u32(entry + 8) : entry + 8);
  const ascii = (entry) => {
    const count = u32(entry + 4), at = valueAt(entry, count);
    let s = "";
    for (let i = 0; i < count - 1; i++) s += String.fromCharCode(view.getUint8(start + at + i));
    return s;
  };
  const rationals = (entry) => {
    const count = u32(entry + 4), at = valueAt(entry, count * 8);
    return Array.from({ length: count }, (_, i) => u32(at + i * 8) / u32(at + i * 8 + 4));
  };

  const ifd0 = readIfd(u32(4));
  const result = { lat: null, lon: null, month: null };

  const exifIfd = ifd0[TAG_EXIF_IFD] ? readIfd(u32(ifd0[TAG_EXIF_IFD] + 8)) : {};
  const dateEntry = exifIfd[TAG_DATE_ORIGINAL] ?? ifd0[TAG_DATE];
  if (dateEntry) {
    const month = Number(ascii(dateEntry).slice(5, 7));
    if (month >= 1 && month <= 12) result.month = month;
  }

  if (ifd0[TAG_GPS_IFD]) {
    const gps = readIfd(u32(ifd0[TAG_GPS_IFD] + 8));
    if (gps[GPS_LAT] && gps[GPS_LON]) {
      const toDegrees = ([d, m, s]) => d + m / 60 + s / 3600;
      let lat = toDegrees(rationals(gps[GPS_LAT]));
      let lon = toDegrees(rationals(gps[GPS_LON]));
      // The ref ("N"/"S", "E"/"W") is stored inline as a 2-byte string
      if (gps[GPS_LAT_REF] && view.getUint8(start + gps[GPS_LAT_REF] + 8) === 0x53) lat = -lat; // "S"
      if (gps[GPS_LON_REF] && view.getUint8(start + gps[GPS_LON_REF] + 8) === 0x57) lon = -lon; // "W"
      // 0,0 usually means "no fix" rather than the Gulf of Guinea
      if (Number.isFinite(lat) && Number.isFinite(lon) && (lat !== 0 || lon !== 0)) {
        Object.assign(result, { lat, lon });
      }
    }
  }
  return result;
}
