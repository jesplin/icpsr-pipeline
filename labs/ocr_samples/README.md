# OCR sample documents

Four public-domain document images used by `labs/11_ocr.py`. All are U.S. federal
records or historical documents in the public domain, retrieved from Wikimedia
Commons. They're chosen to span the cases where OCR difficulty actually lives:
engrossed script, cursive handwriting, and a handwritten administrative table.

| File | What it is | Why it's here |
|------|-----------|---------------|
| `cuban_missile.jpg` | 21 Oct 1962 memo, "Soviet Military Build-Up in Cuba" | Relatively clean typescript w/ outline structure. Tesseract does decently. |
| `cuban_missile2.jpg` | 18 Oct 1962 CIA memo on Mission 3102 | Badly degraded photocopy w/ redaction bars. Tesseract collapses; VLM recovers it (but misreads place names / coordinates). |
| `typescript_1961_letter.jpg` | 1961 National Park Service letter (NARA) | Scanned *typewritten* photocopy: faded, skewed, speckled. Non-intel comparison. |
| `printed_table_pike.jpg` | 1870 census statistical table, Pike County, MO | A *printed table* with a numeric totals row (built-in answer key for validation). |
| `declaration_1776.jpg` | 1823 Stone engraving of the Declaration of Independence | Ornate 18th-century engrossed script + signatures. Printed *title*, handwritten-style *body*. |
| `gettysburg_handwritten.jpg` | Nicolay copy of the Gettysburg Address, in Lincoln's hand | Real cursive handwriting. |
| `lincoln_letter_handwritten.jpg` | Lincoln letter to Mr. Brayman, 1854 (NARA 192847) | A handwritten letter, the everyday historical-archive case. |
| `census_1940_table.jpg` | 1940 U.S. Census population schedule, Sibley County, MN | Handwritten tabular administrative form: the hardest case (handwriting + grid). |

Sources (all public domain):
- Cuban Missile Crisis memos: declassified U.S. intelligence documents (Oct 1962), added by Andy. (Public-domain U.S. government records; original scans from the declassified-document archives.)
- 1961 NPS letter: commons.wikimedia.org (NPS/NARA administrative correspondence, "Director Wirth to Mrs. Gertrude Kibler, August 11, 1961")
- 1870 Pike County table: commons.wikimedia.org (David Rumsey Map Collection, "Statistics of the Population of Pike County by Precincts")
- Declaration of Independence: commons.wikimedia.org/wiki/File:United_States_Declaration_of_Independence.jpg
- Gettysburg Address (Nicolay copy): commons.wikimedia.org/wiki/File:Nicolaycopy.jpg
- Lincoln letter: commons.wikimedia.org/wiki/File:Abraham_Lincoln_Letter_to_Mr._Brayman_September_23,_1854_-_NARA_-_192847.jpg
- 1940 Census schedule: from the DPLA 1940 Census collection on Wikimedia Commons

Images were downloaded at ~1100–1280px on the long edge, which is a reasonable
resolution for OCR (enough detail to read, small enough to keep API token costs down).
