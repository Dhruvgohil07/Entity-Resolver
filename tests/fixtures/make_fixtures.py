"""Generate the tiny Abt-Buy fixtures used by tests/test_data_abt_buy.py.

Written as bytes on purpose: Abt.csv must really be cp1252 so the loader's
encoding handling is exercised rather than assumed. Kept as a script so the
fixtures can be regenerated if the shape they cover ever needs to change.
"""

import pathlib

OUT = pathlib.Path("tests/fixtures/abt-buy")
OUT.mkdir(parents=True, exist_ok=True)

# id 10 collides with Buy id 10 on purpose -- unrelated products, which is
# what forces the side into record_id.
# Row 20 carries the thousands-comma price; row 30 is a singleton with an
# empty price. The (R) in row 10 is the cp1252 byte 0xae.
abt = (
    '"id","name","description","price"\r\n'
    '10,"Sony Turntable - PSLX350H","Belt Drive System® 33-1/3 RPM","$399.00"\r\n'
    '20,"Sharp 32\' LCD TV - LC32D44U","32\' Class Widescreen","$1,299.00"\r\n'
    '30,"Apex Digital DVD Player - AD1500","No price on this row",\r\n'
)

# Buy id 10 is a different product from Abt id 10. Row 55 has a blank
# manufacturer AND a blank description -- both "column present, value empty",
# which must survive as "" and not collapse to None. Rows 98 and 99 are two
# Buy listings of the same turntable, which is what builds a size-3 cluster.
buy = (
    '"id","name","description","manufacturer","price"\r\n'
    '10,"Canon PowerShot Camera - SD1100","12MP compact","Canon","$249.99"\r\n'
    '55,"Sharp LC32D44U 32 inch LCD Television","","","$1,275.00"\r\n'
    '98,"Sony PSLX350H Turntable Belt Drive","Second listing","Sony","$389.00"\r\n'
    '99,"Sony PS-LX350H Belt Drive Turntable","Duplicate listing","SONY",\r\n'
)

# Abt 10 matches both Buy 98 and Buy 99, so the union-find has to merge them
# transitively into one size-3 entity -- the construction that would chain if
# the ground truth were noisy. Abt 20 -> Buy 55 is the ordinary size-2 case.
# Abt 30 and Buy 10 stay unmatched, covering the singleton branch.
mapping = '"idAbt","idBuy"\r\n10,98\r\n10,99\r\n20,55\r\n'

(OUT / "Abt.csv").write_bytes(abt.encode("cp1252"))
(OUT / "Buy.csv").write_bytes(buy.encode("ascii"))
(OUT / "abt_buy_perfectMapping.csv").write_bytes(mapping.encode("ascii"))

for path in sorted(OUT.iterdir()):
    raw = path.read_bytes()
    high = sorted({b for b in raw if b > 127})
    print(f"{path.name:32} {len(raw):5} bytes  high bytes: {[hex(b) for b in high] or 'none'}")
