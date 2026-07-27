const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, BorderStyle, AlignmentType, PageBreak, ShadingType, TabStopType,
  PositionalTab, PositionalTabAlignment, PositionalTabLeader, LevelFormat,
  Header, Footer, PageNumber, NumberFormat, VerticalAlign,
} = require("docx");

// ---------- design tokens ----------
const FONT_BODY = "Calibri";
const FONT_HEAD = "Cambria";
const COL_ACCENT = "1F4E5F";   // deep teal-navy
const COL_ACCENT2 = "B0421A";  // cricket-ball red, used sparingly
const COL_TEXT = "222222";
const COL_MUTE = "5A5A5A";
const COL_RULE = "C9C2B4";
const COL_TABLE_HEAD_BG = "1F4E5F";
const COL_TABLE_ALT_BG = "F2EFE7";
const PAGE_W = 12240, PAGE_H = 15840; // US Letter DXA

// ---------- small helpers ----------
function h1(text, opts = {}) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 280, after: 120 },
    pageBreakBefore: !!opts.pageBreak,
    children: [new TextRun({ text, bold: true, color: COL_ACCENT, font: FONT_HEAD, size: 28 })],
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 180, after: 80 },
    children: [new TextRun({ text, bold: true, color: COL_ACCENT, font: FONT_HEAD, size: 23 })],
  });
}
function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 140, after: 60 },
    children: [new TextRun({ text, bold: true, color: COL_TEXT, font: FONT_HEAD, size: 21 })],
  });
}
function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 110, line: 254 },
    alignment: opts.center ? AlignmentType.CENTER : AlignmentType.JUSTIFIED,
    children: [new TextRun({ text, font: FONT_BODY, size: 20, italics: !!opts.italic, color: opts.color || COL_TEXT })],
  });
}
function pr(runs, opts = {}) {
  return new Paragraph({ spacing: { after: 110, line: 254 }, alignment: opts.center ? AlignmentType.CENTER : AlignmentType.JUSTIFIED, children: runs });
}
function run(text, opts = {}) {
  return new TextRun({ text, font: FONT_BODY, size: opts.size || 20, bold: !!opts.bold, italics: !!opts.italic, color: opts.color || COL_TEXT });
}
function bullet(text, opts = {}) {
  return new Paragraph({
    numbering: { reference: "bullet-list", level: opts.level || 0 },
    spacing: { after: 60, line: 246 },
    children: [new TextRun({ text, font: FONT_BODY, size: 20, color: COL_TEXT })],
  });
}
function reqItem(id, status, text) {
  const statusColor = status.startsWith("Implemented") ? "2E6B34" : status.startsWith("Partially") ? "9C7A12" : status.startsWith("Planned") ? COL_ACCENT2 : COL_MUTE;
  return new Paragraph({
    spacing: { after: 90, line: 250 },
    children: [
      new TextRun({ text: `${id}  `, bold: true, font: FONT_BODY, size: 20, color: COL_ACCENT }),
      new TextRun({ text: `[${status}]  `, bold: true, italics: true, font: FONT_BODY, size: 18, color: statusColor }),
      new TextRun({ text, font: FONT_BODY, size: 20, color: COL_TEXT }),
    ],
  });
}
function hr() {
  return new Paragraph({
    spacing: { before: 60, after: 200 },
    border: { bottom: { color: COL_RULE, space: 1, style: BorderStyle.SINGLE, size: 6 } },
    children: [new TextRun({ text: "" })],
  });
}
function cell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.width || 2000, type: WidthType.DXA },
    shading: opts.head ? { type: ShadingType.CLEAR, fill: COL_TABLE_HEAD_BG } : opts.alt ? { type: ShadingType.CLEAR, fill: COL_TABLE_ALT_BG } : undefined,
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 50, bottom: 50, left: 100, right: 100 },
    children: [new Paragraph({
      spacing: { line: 240 },
      alignment: opts.center ? AlignmentType.CENTER : AlignmentType.LEFT,
      children: [new TextRun({ text: String(text), bold: !!opts.head, color: opts.head ? "FFFFFF" : COL_TEXT, font: FONT_BODY, size: opts.size || 18 })],
    })],
  });
}
function table(headers, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, { head: true, width: widths[i], center: i > 0 })) }),
      ...rows.map((r, ri) => new TableRow({ children: r.map((c, i) => cell(c, { alt: ri % 2 === 1, width: widths[i], center: i > 0 })) })),
    ],
  });
}
function caption(text) {
  return new Paragraph({ spacing: { before: 80, after: 240 }, children: [new TextRun({ text, italics: true, size: 18, color: COL_MUTE, font: FONT_BODY })] });
}

module.exports = { fs, Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType, BorderStyle, AlignmentType, PageBreak, ShadingType, VerticalAlign, PAGE_W, PAGE_H, FONT_BODY, FONT_HEAD, COL_ACCENT, COL_ACCENT2, COL_TEXT, COL_MUTE, COL_RULE, h1, h2, h3, p, pr, run, bullet, reqItem, hr, cell, table, caption };
