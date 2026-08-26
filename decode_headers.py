"""Properly decode the Cyrillic headers using char code mapping."""

import xlrd

EXCEL_PATH = r"C:\1001110001000111101(1)\Python\Test_metal\source_data.xls"
book = xlrd.open_workbook(EXCEL_PATH, formatting_info=False)
sheet = book.sheet_by_index(0)
header_row = 3

# Decode each char: if it's in Cyrillic range, it's already correct Unicode
# The xlrd returns proper Unicode strings, the "garbled" display is a cp1252 console issue
cyrillic_map = {
    0x0410: "A",
    0x0411: "B",
    0x0412: "V",
    0x0413: "G",
    0x0414: "D",
    0x0415: "E",
    0x0416: "ZH",
    0x0417: "Z",
    0x0418: "I",
    0x0419: "J",
    0x041A: "K",
    0x041B: "L",
    0x041C: "M",
    0x041D: "N",
    0x041E: "O",
    0x041F: "P",
    0x0420: "R",
    0x0421: "S",
    0x0422: "T",
    0x0423: "U",
    0x0424: "F",
    0x0425: "H",
    0x0426: "TS",
    0x0427: "CH",
    0x0428: "SH",
    0x0429: "SHCH",
    0x042A: "",
    0x042B: "Y",
    0x042C: "",
    0x042D: "E",
    0x042E: "YU",
    0x042F: "YA",
    0x0430: "a",
    0x0431: "b",
    0x0432: "v",
    0x0433: "g",
    0x0434: "d",
    0x0435: "e",
    0x0436: "zh",
    0x0437: "z",
    0x0438: "i",
    0x0439: "j",
    0x043A: "k",
    0x043B: "l",
    0x043C: "m",
    0x043D: "n",
    0x043E: "o",
    0x043F: "p",
    0x0440: "r",
    0x0441: "s",
    0x0442: "t",
    0x0443: "u",
    0x0444: "f",
    0x0445: "h",
    0x0446: "ts",
    0x0447: "ch",
    0x0448: "sh",
    0x0449: "shch",
    0x044A: "",
    0x044B: "y",
    0x044C: "",
    0x044D: "e",
    0x044E: "yu",
    0x044F: "ya",
    0x00A0: " ",
    0x2116: "No.",
}


def transliterate(s):
    result = []
    for c in s:
        if ord(c) in cyrillic_map:
            result.append(cyrillic_map[ord(c)])
        elif ord(c) > 127:
            result.append(f"<U+{ord(c):04X}>")
        else:
            result.append(c)
    return "".join(result)


print("TRANSLITERATED headers for all 91 positions:")
print()
for col in range(91):
    cell = sheet.cell_value(header_row, col + 1)
    trans = transliterate(cell) if isinstance(cell, str) else str(cell)
    print(f"  [{col:>2}] {trans[:90]}")
