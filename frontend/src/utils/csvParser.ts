/**
 * Minimal CSV parser that handles quoted fields and returns typed row arrays.
 * Assumes UTF-8 encoding and Unix/Windows line endings.
 */
export function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = '';
  let inQuote = false;

  for (let i = 0; i < text.length; i++) {
    const ch = text[i];

    if (inQuote) {
      if (ch === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; }
        else inQuote = false;
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      inQuote = true;
    } else if (ch === ',') {
      row.push(field);
      field = '';
    } else if (ch === '\n') {
      row.push(field);
      field = '';
      if (row.some(f => f !== '')) rows.push(row);
      row = [];
    } else if (ch !== '\r') {
      field += ch;
    }
  }

  // flush last field/row
  if (field !== '' || row.length > 0) {
    row.push(field);
    if (row.some(f => f !== '')) rows.push(row);
  }

  return rows;
}

/**
 * Parse CSV text into records keyed by the header row.
 */
export function parseCsvRecords(text: string): Record<string, string>[] {
  const rows = parseCsv(text);
  if (rows.length < 2) return [];
  const [header, ...dataRows] = rows;
  return dataRows.map(row =>
    Object.fromEntries(header.map((key, i) => [key.trim(), (row[i] ?? '').trim()]))
  );
}
