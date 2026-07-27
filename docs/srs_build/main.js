const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, WidthType, BorderStyle,
  AlignmentType,
  PAGE_W, PAGE_H, FONT_BODY, FONT_HEAD, COL_ACCENT, COL_MUTE, COL_RULE,
} = require("./gen_srs.js");
const { LevelFormat, PageNumber, Header, Footer } = require("docx");
const fs = require("fs");
const { children } = require("./build_doc.js");

const doc = new Document({
  creator: "Kurapati Sai Suhas",
  title: "SuhasVision — Software Requirements Specification v2.0",
  numbering: {
    config: [
      {
        reference: "bullet-list",
        levels: [
          { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 460, hanging: 260 } } } },
          { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 820, hanging: 260 } } } },
        ],
      },
    ],
  },
  styles: {
    default: {
      document: { run: { font: FONT_BODY, size: 21 } },
      heading1: { run: { font: FONT_HEAD, bold: true, color: COL_ACCENT, size: 30 }, paragraph: { spacing: { before: 480, after: 200 } } },
      heading2: { run: { font: FONT_HEAD, bold: true, color: COL_ACCENT, size: 24 }, paragraph: { spacing: { before: 320, after: 140 } } },
      heading3: { run: { font: FONT_HEAD, bold: true, color: "222222", size: 21 }, paragraph: { spacing: { before: 220, after: 100 } } },
    },
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: PAGE_W, height: PAGE_H },
          margin: { top: 1080, bottom: 1080, left: 1260, right: 1260 },
        },
      },
      headers: {
        default: new Header({
          children: [new Paragraph({
            border: { bottom: { color: COL_RULE, size: 4, style: BorderStyle.SINGLE, space: 4 } },
            children: [
              new TextRun({ text: "SuhasVision — Software Requirements Specification v2.0", size: 15, color: COL_MUTE, font: FONT_BODY, italics: true }),
            ],
          })],
        }),
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [
              new TextRun({ text: "Page ", size: 16, color: COL_MUTE, font: FONT_BODY }),
              new TextRun({ children: [PageNumber.CURRENT], size: 16, color: COL_MUTE, font: FONT_BODY }),
              new TextRun({ text: " of ", size: 16, color: COL_MUTE, font: FONT_BODY }),
              new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 16, color: COL_MUTE, font: FONT_BODY }),
            ],
          })],
        }),
      },
      children,
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("./SuhasVision_SRS_v2.0.docx", buf);
  console.log("Wrote SuhasVision_SRS_v2.0.docx,", buf.length, "bytes");
});
