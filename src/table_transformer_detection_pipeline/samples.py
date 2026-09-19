"""Box-labelled image dataset contract for adapting the detector: the pinned Open Food Facts nutrition-table
sample, validation, seeded splitting, BYOD loaders and CSV export.

The default dataset is **real** and out of the checkpoint's document-page domain: 121 product photographs
from the Open Food Facts nutrition-table detection set (version 1.1, manually reviewed boxes; images CC BY-
SA 3.0, one to three nutrition-table boxes each), pinned here per file by byte size and SHA-256 of the
served original on `static.openfoodfacts.org` — the served originals were checked against the labelled
images (same size, no EXIF rotation, pixel MAE < 1/255), and the two whose served original carries an EXIF
orientation tag were left out of the 123-image split. The normalised boxes travel with each record. Every
file is fetched at run time and refused on any byte-size or SHA-256 mismatch, then downscaled to a longest
side of 1,280 px (the originals reach 5,312 px; the detector's processor resizes to 800 px anyway); the
repository redistributes none of the photographs, and each record keeps its barcode and image URL. Two
products have two photographs each, so the sample is split **by product**, never by photograph.

A record is ``{id, image, boxes}``: a PIL image (or a path to one) and a list of ``[x_min, y_min, x_max,
y_max]`` pixel boxes of the tables on it, all of one class (the four Open Food Facts nutrition-table
categories are merged into one `nutrition-table` class and kept under ``categories`` for provenance).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from .pipeline import MAX_DETECTIONS, MAX_IMAGE_SIDE, MIN_IMAGE_SIDE, MODEL_ID

CORPUS_NAME = "Open Food Facts nutrition-table photographs"
CORPUS_RELEASE = (
    "openfoodfacts/nutrition-table-detection v1.1 validation split (Hugging Face Hub commit d59dcee8), "
    "served originals pinned 2026-09-19"
)
CORPUS_BASE_URL = "https://static.openfoodfacts.org/images/products/"
CORPUS_LICENSE = "CC BY-SA 3.0 (Open Food Facts images; boxes from the Open Food Facts dataset, ODbL)"
CORPUS_BYTES = 129_318_620
CLASS_NAME = "nutrition-table"
CORPUS_LONGEST_SIDE = 1280  # served originals (up to 5,312 px) are downscaled to this at read time
SKIPPED_EXIF = ("0024138012322_3", "0041449003153_2")  # served originals with an EXIF orientation tag

# (record id, barcode, URL path under CORPUS_BASE_URL, width, height, bytes, sha256,
#  "ymin,xmin,ymax,xmax;..." normalised boxes, "category;..." per box)
SAMPLE_RECORDS: tuple[tuple[str, str, str, int, int, int, str, str, str], ...] = (
    (
        "0041129077641_1",
        "0041129077641",
        "004/112/907/7641/1.jpg",
        3120,
        4208,
        1998704,
        "d62e1721d6e1110fa129750a3718f0e8198abadc03801a0237ddbf986197e0d0",
        "0.487167,0.273397,0.640934,0.730449",
        "nutrition-table",
    ),
    (
        "0070200581159_1",
        "0070200581159",
        "007/020/058/1159/1.jpg",
        2448,
        3264,
        1657683,
        "adebe42a7b4d662ca5c1265eac6808ef308d66c58d5b6b3ceb4ce162b2b19e67",
        "0.246630,0.265523,0.453257,0.782555",
        "nutrition-table",
    ),
    (
        "0064144043064_2",
        "0064144043064",
        "006/414/404/3064/2.jpg",
        2000,
        2666,
        566828,
        "9608bb7e5ba84ef1efcc034cd01cbb517fb0d81b53ce9fc5c9fdf787588c3c52",
        "0.103151,0.040950,0.514332,0.938593",
        "nutrition-table",
    ),
    (
        "0051500700167_1",
        "0051500700167",
        "005/150/070/0167/1.jpg",
        2448,
        3264,
        719315,
        "78ada22c3c2f97f3808c47b591b91767d9b37c26d977fa0b04fc470fb62c20d5",
        "0.314339,0.164324,0.833649,0.584703",
        "nutrition-table",
    ),
    (
        "0016447100951_1",
        "0016447100951",
        "001/644/710/0951/1.jpg",
        2000,
        2697,
        460770,
        "1897c94d877f14f53c5388fc5060271a3ca2fb1d018a9cc4444a84cbfd3c68f1",
        "0.362996,0.350000,0.615499,0.559500",
        "nutrition-table",
    ),
    (
        "26195070_4",
        "26195070",
        "26195070/4.jpg",
        2000,
        1125,
        191335,
        "5555ecda7daf2ded79717034567ed6058aeb8461d1bbf4486cc1d4b64611e9e1",
        "0.144000,0.318864,0.807477,0.702636",
        "nutrition-table",
    ),
    (
        "0041244641024_3",
        "0041244641024",
        "004/124/464/1024/3.jpg",
        1675,
        1200,
        433717,
        "0d46658a1de8ed3f6facf5c2eb9ba1be56c45f65a43c5ea858bad080a96dcbdf",
        "0.000000,0.004179,1.000000,1.000000",
        "nutrition-table",
    ),
    (
        "26235066_5",
        "26235066",
        "26235066/5.jpg",
        1125,
        2000,
        435248,
        "d31bc4e03ea07556964d647c9d1b5292a29ba64e97b23ce47c16cbbf46dcb533",
        "0.442000,0.227556,0.793458,0.854545",
        "nutrition-table",
    ),
    (
        "3571492670004_7",
        "3571492670004",
        "357/149/267/0004/7.jpg",
        953,
        1200,
        243285,
        "125ec3aa086eee041192289492367feca403bd01cc6a27056ad317dcd7064394",
        "0.013333,0.013641,0.620000,0.956978",
        "nutrition-table",
    ),
    (
        "3256224806950_3",
        "3256224806950",
        "325/622/480/6950/3.jpg",
        788,
        491,
        110197,
        "8eea64b979e2cd300adc4ced0ecde024d9fef6687cf54c44db5abc18ef3852d5",
        "0.260692,0.040609,0.977597,0.569797",
        "nutrition-table",
    ),
    (
        "0072417153051_3",
        "0072417153051",
        "007/241/715/3051/3.jpg",
        1944,
        2592,
        1929918,
        "46c641c50efa7995e86400d34b0023d0da45070d24ce69f259ca85fad30a844a",
        "0.298586,0.266057,0.828701,0.701383;0.823117,0.395089,0.999878,0.671329",
        "nutrition-table;nutrition-table-small",
    ),
    (
        "103106_4",
        "103106",
        "103106/4.jpg",
        2434,
        1817,
        880404,
        "83700c169a4ccf16a5c7db0e71fb4fe60d67313f5655d4cd6233238580c01710",
        "0.000000,0.000000,0.999901,1.000000",
        "nutrition-table",
    ),
    (
        "20635923_2",
        "20635923",
        "20635923/2.jpg",
        2448,
        3264,
        1102005,
        "55a2b82ad9d1211a3f91d2bb737eb6a7e803923afd8af5e26968eccbbe57b679",
        "0.095565,0.558065,0.854032,0.682258",
        "nutrition-table-text",
    ),
    (
        "0072417159152_2",
        "0072417159152",
        "007/241/715/9152/2.jpg",
        2448,
        3264,
        1248922,
        "3d7eb73f60b877a2940949c85af49b711cfb69764859a3738e7f6c9195f457bf",
        "0.389458,0.120351,0.920936,0.569505;0.471405,0.925245,0.758987,1.000000;0.923077,0.240559,1.000000,0.527739",
        "nutrition-table;nutrition-table-small-energy;nutrition-table-small",
    ),
    (
        "3250390768296_4",
        "3250390768296",
        "325/039/076/8296/4.jpg",
        2000,
        1500,
        972649,
        "e37e975192ba1d8f049d2e96ba786b0bc1cf459f64ae163a67ce1135314e42d4",
        "0.532000,0.862000,0.653333,0.903500;0.647333,0.860500,0.680000,0.893500",
        "nutrition-table-small-energy;nutrition-table-small",
    ),
    (
        "0071479000013_3",
        "0071479000013",
        "007/147/900/0013/3.jpg",
        3264,
        2448,
        1845181,
        "24d5a091c551aa01404d21343529dad0e20fb3d987d6747ce9b991dc0c93b7d9",
        "0.466314,0.163282,0.886736,0.584477",
        "nutrition-table",
    ),
    (
        "0011225033674_1",
        "0011225033674",
        "001/122/503/3674/1.jpg",
        3264,
        2448,
        1825039,
        "c92ce4dcde826aed05d1d9aa814855f7c12fa8bb0d3c1e5d8504dd7db7af4697",
        "0.241830,0.004132,0.668496,0.943321",
        "nutrition-table",
    ),
    (
        "7610700946053_2",
        "7610700946053",
        "761/070/094/6053/2.jpg",
        1020,
        1360,
        346264,
        "10ace5b6fca88d180b10e01f9cc038af97cdf4480a3148203466be1f9c336388",
        "0.725000,0.275490,0.810294,0.951961",
        "nutrition-table-text",
    ),
    (
        "0064100108219_3",
        "0064100108219",
        "006/410/010/8219/3.jpg",
        2000,
        2666,
        613190,
        "b0395c01a4b58794d958f03fd1567fff7974531a1aa2d1c483197048da156300",
        "0.138785,0.179000,0.620405,0.652000",
        "nutrition-table",
    ),
    (
        "0073416006102_2",
        "0073416006102",
        "007/341/600/6102/2.jpg",
        2000,
        2666,
        414784,
        "d09cf4a32cb1a8ad1c9ae9d73a08752e9acd1a7695d3e1ce9079af9ad5ddd7e5",
        "0.225056,0.270500,0.559265,0.572500",
        "nutrition-table",
    ),
    (
        "26058450_2",
        "26058450",
        "26058450/2.jpg",
        819,
        650,
        207112,
        "e54e724b2d4c880a357b0afb927ecbf2d97b721a2e4a9ec43d9ea1f5ebf9ec8a",
        "0.000000,0.000000,1.000000,1.000000",
        "nutrition-table",
    ),
    (
        "27429464_2",
        "27429464",
        "27429464/2.jpg",
        1944,
        2592,
        1593330,
        "81ec8d7ecc5e647f01b495a96af3f3dc14b6c94d12a10bbe6eef4a61c4eb6a2a",
        "0.525849,0.172196,0.906347,0.480646",
        "nutrition-table",
    ),
    (
        "26052656_1",
        "26052656",
        "26052656/1.jpg",
        2000,
        3555,
        540007,
        "48210c9119f367bc418bd5de79aff2cab7f7cb0f24057eee140e1b330dbc8817",
        "0.776090,0.111000,0.810127,0.157000",
        "nutrition-table-small-energy",
    ),
    (
        "26036120_5",
        "26036120",
        "26036120/5.jpg",
        1560,
        2000,
        1226507,
        "b4a53be06ed5eb8f7f21471f764248ab653afe38d5c5e8f03bd264f388b18b04",
        "0.000000,0.019231,0.748500,0.988462",
        "nutrition-table",
    ),
    (
        "0042272005420_2",
        "0042272005420",
        "004/227/200/5420/2.jpg",
        1598,
        1360,
        356803,
        "46fc64d0083a6b72665a329c511720efd9c47c60f39835ace4bbbf325041dfef",
        "0.045284,0.010638,0.949277,0.995382",
        "nutrition-table",
    ),
    (
        "26191225_2",
        "26191225",
        "26191225/2.jpg",
        1125,
        2000,
        518361,
        "0bd0b5ee0e01250e6f1b4c1200d08f0d1cc0867eaafb94ae96269ac4bdd85702",
        "0.090717,0.035556,0.645976,0.874667",
        "nutrition-table",
    ),
    (
        "3250391868322_1",
        "3250391868322",
        "325/039/186/8322/1.jpg",
        2000,
        3561,
        632711,
        "3f059fce232b33e1bb3e58cc8ea80e2fcc540558cf87fa1ab3c93935fe188e63",
        "0.411401,0.083000,0.437237,0.550500",
        "nutrition-table-text",
    ),
    (
        "3222472951582_1",
        "3222472951582",
        "322/247/295/1582/1.jpg",
        3264,
        2448,
        2621579,
        "6ebf0edf2b8cfac7b3eec51a6465863e0aa7cb7b1d3ef25b6d70e734cb823e65",
        "0.403595,0.157169,0.502859,0.840380",
        "nutrition-table-text",
    ),
    (
        "0028400160148_3",
        "0028400160148",
        "002/840/016/0148/3.jpg",
        3120,
        4208,
        1852904,
        "03f3ba30d924d73f92fa101ff303fce02338bfb3dfd7d017b6db47a6e9b09751",
        "0.078422,0.260577,0.524715,0.694231",
        "nutrition-table",
    ),
    (
        "50172436_2",
        "50172436",
        "50172436/2.jpg",
        2000,
        2666,
        855873,
        "3d19028b2efbdbf8657e5e9337a589e42730b835389e8fbd8d1d01438a3a8644",
        "0.518545,0.247500,0.814430,0.753864",
        "nutrition-table",
    ),
    (
        "3564700435106_1",
        "3564700435106",
        "356/470/043/5106/1.jpg",
        1500,
        2000,
        1097943,
        "fde2ef2a5d4e7567c10dcc30228cbbead119df09ce151c9fe341fdf07efdeb30",
        "0.455000,0.033333,0.784000,0.768000",
        "nutrition-table",
    ),
    (
        "0070852000527_3",
        "0070852000527",
        "007/085/200/0527/3.jpg",
        3264,
        2448,
        1690640,
        "071428dce8f777aa61e1b3e23f613be6df7d4bf33ac61a071dc32bfd980b76e2",
        "0.300670,0.266831,0.680981,0.564658",
        "nutrition-table",
    ),
    (
        "0074333375531_2",
        "0074333375531",
        "007/433/337/5531/2.jpg",
        1125,
        2000,
        419124,
        "38210cb65b6aa213a460b9d8586b91e464bef30e02b76b192424f06e738cd89e",
        "0.015500,0.327111,0.421297,0.842810",
        "nutrition-table",
    ),
    (
        "0028400071345_2",
        "0028400071345",
        "002/840/007/1345/2.jpg",
        3264,
        2448,
        1838123,
        "40843cd14c1c274610539a458b0445c5580ebbefac377e673a7d52cf1d654b31",
        "0.301517,0.416045,0.730426,0.871762",
        "nutrition-table",
    ),
    (
        "20720162_2",
        "20720162",
        "20720162/2.jpg",
        2448,
        3264,
        1138660,
        "462439a74d913dab460af652daec4ac467696d433f864fabf5836dab1d3d2552",
        "0.122243,0.228119,0.764797,0.738544",
        "nutrition-table",
    ),
    (
        "3700003780349_1",
        "3700003780349",
        "370/000/378/0349/1.jpg",
        3087,
        2488,
        2204312,
        "a6f360b4aa545aa6739016eb0895e0aca940cceb3ad41645685088db04876e8d",
        "0.729904,0.188209,0.806270,0.796242",
        "nutrition-table-text",
    ),
    (
        "0037600106252_4",
        "0037600106252",
        "003/760/010/6252/4.jpg",
        3120,
        4208,
        2362228,
        "e0ef8fc6bca8918408425a4b3f88e2cf81379e555b5200eca7a9e801b0fc85e3",
        "0.249525,0.245513,0.668617,0.533974",
        "nutrition-table",
    ),
    (
        "0070970471254_2",
        "0070970471254",
        "007/097/047/1254/2.jpg",
        1125,
        2000,
        524563,
        "b001b96b7e0421ea6a3decf3687b0b23ecf5aae117b7b1c64e4bc0b0bb084e1f",
        "0.485918,0.375603,0.648502,0.798635;0.255484,0.159605,0.449773,0.320211",
        "nutrition-table;nutrition-table-small",
    ),
    (
        "01575118_2",
        "01575118",
        "01575118/2.jpg",
        1125,
        2000,
        468903,
        "fe5cf31b1e64d84c609c61322e328f4b7f9a6118fa5b260084e125445acea752",
        "0.315541,0.350398,0.636206,0.544194",
        "nutrition-table-text",
    ),
    (
        "0016000264694_1",
        "0016000264694",
        "001/600/026/4694/1.jpg",
        2000,
        1500,
        285622,
        "c8e0eb613ea129586d33aae5d00146d78bdce2943fe36f08e42f4859fe54ac44",
        "0.386667,0.023500,0.682000,0.768500",
        "nutrition-table-text",
    ),
    (
        "0021000653218_1",
        "0021000653218",
        "002/100/065/3218/1.jpg",
        455,
        2000,
        644198,
        "dc5d734f4f99b840312d6cbdcdd35a1f52493c2eeae4fa0349682fb0afabb5ec",
        "0.023000,0.120879,0.450500,0.835165",
        "nutrition-table",
    ),
    (
        "20840822_1",
        "20840822",
        "20840822/1.jpg",
        1125,
        2000,
        410087,
        "8446d47854e21fdeaca9d4a1ccfbd2c2dc256412a7eb1a78acb57973244c4abc",
        "0.468000,0.039111,0.733500,0.480889",
        "nutrition-table",
    ),
    (
        "0011156054502_2",
        "0011156054502",
        "001/115/605/4502/2.jpg",
        2448,
        3264,
        1131895,
        "ac9afee8bef22768404ca119827dc03c892e80ab3f1f7002164b56da7c205031",
        "0.059436,0.091912,0.639400,0.860294",
        "nutrition-table",
    ),
    (
        "0016000275348_2",
        "0016000275348",
        "001/600/027/5348/2.jpg",
        1125,
        2000,
        511106,
        "e3aa8bbf1e2abd4b615dda312f1dcebda2c57f537c0397a873405bae653007db",
        "0.058000,0.396444,0.365000,0.684444",
        "nutrition-table",
    ),
    (
        "0046100001639_2",
        "0046100001639",
        "004/610/000/1639/2.jpg",
        2988,
        5312,
        5173381,
        "4aa8d0adb67c50d695fdb2ce096d94570907c2ef0537f742641c4ee4e2a47f72",
        "0.203502,0.310241,0.534317,0.751004",
        "nutrition-table",
    ),
    (
        "0018627703211_3",
        "0018627703211",
        "001/862/770/3211/3.jpg",
        332,
        1360,
        150374,
        "f47ade12d2fbdd77b3e0b73afdb84a26a9dcc0cf35939ac623ec6d41453d4c35",
        "0.181618,0.036145,0.536029,0.972892",
        "nutrition-table",
    ),
    (
        "0073007107140_1",
        "0073007107140",
        "007/300/710/7140/1.jpg",
        3120,
        4208,
        2921589,
        "b8683ed21fbdc4f22c23885fad16f565987340b34e2a358e72d1c7b7c08343d9",
        "0.460314,0.514744,0.677281,0.749680",
        "nutrition-table",
    ),
    (
        "0036200013694_2",
        "0036200013694",
        "003/620/001/3694/2.jpg",
        3120,
        4208,
        1553624,
        "badbd8fa1eeacfea9b9c642af8d8a204d69c2485828b85fca3b9d9e39a214dd7",
        "0.360796,0.360897,0.760877,0.656510",
        "nutrition-table",
    ),
    (
        "26191218_2",
        "26191218",
        "26191218/2.jpg",
        1125,
        2000,
        533525,
        "34d180f661e918022fad563761697cee4e629b1ad6b2f6e544943302d9fef73d",
        "0.149210,0.072000,0.673280,0.852444",
        "nutrition-table",
    ),
    (
        "20574369_2",
        "20574369",
        "20574369/2.jpg",
        640,
        640,
        184489,
        "c3513d461d0eaf6f54d10fc1781187a1fb9bc428bd2caa1e2c1e2b70aaa2243b",
        "0.621875,0.510938,0.751562,0.851562",
        "nutrition-table-small",
    ),
    (
        "26167932_4",
        "26167932",
        "26167932/4.jpg",
        880,
        738,
        239860,
        "3c74bc09b35fb62fc4ef399b183b809ea6020f5366a8bcda1b9054bb56daf7d0",
        "0.000000,0.006267,1.000000,0.992011",
        "nutrition-table",
    ),
    (
        "0049000027624_2",
        "0049000027624",
        "004/900/002/7624/2.jpg",
        2000,
        1500,
        360138,
        "9f0050076a6f94d02b7fdfad867c5b5589deeae5d6d95d0a4f642f9412cd3f2c",
        "0.157470,0.032597,0.862847,0.754699",
        "nutrition-table",
    ),
    (
        "20574444_2",
        "20574444",
        "20574444/2.jpg",
        2592,
        1936,
        661531,
        "f887b9c596059aa276bd6cba09524d0d63341b23c5b524d4cfffa633d9689878",
        "0.415806,0.353009,0.770295,0.739969",
        "nutrition-table",
    ),
    (
        "0038000787270_2",
        "0038000787270",
        "003/800/078/7270/2.jpg",
        2448,
        3264,
        1019308,
        "ee129674479faacb3248212111ab1013d87960ee3178622a322745c9b16d51af",
        "0.067416,0.243873,0.582108,0.585784",
        "nutrition-table",
    ),
    (
        "0027400264993_1",
        "0027400264993",
        "002/740/026/4993/1.jpg",
        3120,
        4208,
        1962742,
        "c1031a435fda0ea9b686b53c5cee4b092f7084e8757c563db8b6a390218de308",
        "0.355038,0.166346,0.629753,0.435577",
        "nutrition-table",
    ),
    (
        "2407968021654_2",
        "2407968021654",
        "240/796/802/1654/2.jpg",
        1911,
        3302,
        1296492,
        "adf7bc130ddfa3c386b9016e8b555e403d660880636e2f26cd58a3f5883144e5",
        "0.320412,0.077446,0.383707,0.769754",
        "nutrition-table-text",
    ),
    (
        "0014054030715_2",
        "0014054030715",
        "001/405/403/0715/2.jpg",
        2000,
        2666,
        451284,
        "95d55fa77fa76ef0fe0ad579941dfdf6bb8fa82fad75e16f5f208dd670fa0fcc",
        "0.149662,0.169000,0.672543,0.740500",
        "nutrition-table",
    ),
    (
        "50300853_1",
        "50300853",
        "50300853/1.jpg",
        1944,
        2592,
        1053994,
        "b2ec37a6a066eafe95c9a9555796331441a2c8391b40cad335805fe099918995",
        "0.712191,0.353909,0.839120,0.484568",
        "nutrition-table",
    ),
    (
        "3250391868322_6",
        "3250391868322",
        "325/039/186/8322/6.jpg",
        1021,
        1360,
        322658,
        "5c07f704349e7d476c0223f29681d44237b495854e0c4a44a3b0344e75a673f9",
        "0.500000,0.061704,0.540441,0.822723",
        "nutrition-table-text",
    ),
    (
        "3410280003832_1",
        "3410280003832",
        "341/028/000/3832/1.jpg",
        3024,
        3270,
        3261686,
        "e986e16842a9f4371cf8cc16041ee742838aceca4c29b6459e9d1432b0bc0cd3",
        "0.509480,0.111772,0.807034,0.697421",
        "nutrition-table",
    ),
    (
        "3230140005024_3",
        "3230140005024",
        "323/014/000/5024/3.jpg",
        3024,
        4032,
        1457937,
        "90c8e5293feee5cc83bacd18f32859ca035b530ab4affcf52c6515f156e9132f",
        "0.410714,0.167328,0.687004,0.753968",
        "nutrition-table",
    ),
    (
        "26067674_3",
        "26067674",
        "26067674/3.jpg",
        2000,
        1125,
        289442,
        "d937845f8dfaafb85cfb1e1cc7d4390355cd1550d584c5e10e5a2227f370747b",
        "0.031111,0.447985,0.638222,0.871908",
        "nutrition-table",
    ),
    (
        "20719159_3",
        "20719159",
        "20719159/3.jpg",
        2448,
        3264,
        856874,
        "02182f1c2f0e9e46172ff73ac54e4a5cffd2c896ec72aaec1b4c739e7bab8877",
        "0.204378,0.177288,0.562938,0.815768;0.516238,0.295343,0.517770,0.297794",
        "nutrition-table;nutrition-table",
    ),
    (
        "0034000123803_7",
        "0034000123803",
        "003/400/012/3803/7.jpg",
        4208,
        3120,
        2263439,
        "235dd635ed23d93e821f3f85dd569953948afad2f2ebfb4d06e697ed2d4ce12c",
        "0.239104,0.314876,0.522044,0.709030",
        "nutrition-table",
    ),
    (
        "3222473615476_3",
        "3222473615476",
        "322/247/361/5476/3.jpg",
        2000,
        1500,
        933034,
        "34ac75f9907f3c53637077142aa10038da22cde27f875631af84454c0829abb6",
        "0.348000,0.058500,0.592000,0.946500",
        "nutrition-table-text",
    ),
    (
        "0067312002832_4",
        "0067312002832",
        "006/731/200/2832/4.jpg",
        1512,
        1533,
        855217,
        "aeedc599789b6add7aefbdbc0439d61e05b9e21944b646499bbf6578c8088edd",
        "0.140900,0.019180,0.789954,0.460317",
        "nutrition-table",
    ),
    (
        "26015637_2",
        "26015637",
        "26015637/2.jpg",
        1125,
        2000,
        451648,
        "ca56eb2bc7df1b6d6428f997cbf405ae7261066bf1b9d0149a5b798d1d226806",
        "0.236070,0.139492,0.833118,0.997645",
        "nutrition-table",
    ),
    (
        "0025616102504_2",
        "0025616102504",
        "002/561/610/2504/2.jpg",
        1125,
        2000,
        562884,
        "84096589220e1c074177915778aec2732555c4dfc8afec793610ac5c0e80fe0d",
        "0.287000,0.080000,0.846255,0.778667",
        "nutrition-table",
    ),
    (
        "20551926_2",
        "20551926",
        "20551926/2.jpg",
        3264,
        2448,
        1246335,
        "09716c32bba4f7b00b77fff227b7f62e78bedee33a8fdabaaf268479be9c4c30",
        "0.191844,0.162990,0.696078,0.815863",
        "nutrition-table",
    ),
    (
        "0054800010080_3",
        "0054800010080",
        "005/480/001/0080/3.jpg",
        3120,
        4208,
        2607798,
        "ed7569bd0aa59da0632c115f6fb90cfca8bc2b02329dbdc4b3bbe15e9cfce49a",
        "0.031743,0.334121,0.436861,0.629377",
        "nutrition-table",
    ),
    (
        "20117795_2",
        "20117795",
        "20117795/2.jpg",
        1970,
        1360,
        655546,
        "6aa3e4bed9fce3b2a9d4622cc33cd479676af70e7f0616e96ffa853b99d64bb7",
        "0.014371,0.020784,0.905147,0.960304",
        "nutrition-table",
    ),
    (
        "0043647020017_2",
        "0043647020017",
        "004/364/702/0017/2.jpg",
        1944,
        2592,
        1084683,
        "c8dc0362a3b3ecf36d3a6536abef370610865cfb3e721acb57e44caf76d866d6",
        "0.589892,0.336934,0.721836,0.606996",
        "nutrition-table",
    ),
    (
        "0043646210389_1",
        "0043646210389",
        "004/364/621/0389/1.jpg",
        3120,
        4208,
        2451229,
        "debbe1e8f795ef6641e3dfd874c64b78b2fd3f3b43ea0e1b74edab6172a9e517",
        "0.518774,0.266987,0.653517,0.681410",
        "nutrition-table-text",
    ),
    (
        "26117959_7",
        "26117959",
        "26117959/7.jpg",
        1363,
        863,
        369552,
        "1eadc776c8df97a0d32d2419681e3a6efac0a0e7dcf121a8db3587dc8d536cec",
        "0.000000,0.000000,1.000000,1.000000",
        "nutrition-table",
    ),
    (
        "20165079_2",
        "20165079",
        "20165079/2.jpg",
        2448,
        3264,
        1246787,
        "a160efe0373f4dcd2b7dc2b3e2507974b5e1d1bb62016dd9ed07fd35987b7232",
        "0.075954,0.050245,0.479565,0.903826",
        "nutrition-table",
    ),
    (
        "24632621_4",
        "24632621",
        "24632621/4.jpg",
        2448,
        3264,
        1817539,
        "0c4e43290d5c21a460acb65e0bc42c3a00a3bd59b007fd26ea88563a1699370c",
        "0.317708,0.046160,0.725490,0.978758",
        "nutrition-table",
    ),
    (
        "9300601250240_2",
        "9300601250240",
        "930/060/125/0240/2.jpg",
        1125,
        2000,
        477012,
        "b1de38fd1279143e3529f65fbd6af8a1efab06f43329fbef24a3f7df9961064e",
        "0.460500,0.102222,0.804500,0.796444",
        "nutrition-table",
    ),
    (
        "20674540_2",
        "20674540",
        "20674540/2.jpg",
        2448,
        3264,
        1051317,
        "273b324618c15cf0b6d0fee954c210221b3c35310d12608a70e945589d49a0c0",
        "0.631913,0.387992,0.846302,0.939461",
        "nutrition-table",
    ),
    (
        "0072878515276_2",
        "0072878515276",
        "007/287/851/5276/2.jpg",
        3120,
        4208,
        1674429,
        "f87f8f37a3035186b8aa684a96d5f72debe84851eb1049467cff1600a5264bb1",
        "0.437975,0.375962,0.736556,0.642555",
        "nutrition-table",
    ),
    (
        "3250390023777_3",
        "3250390023777",
        "325/039/002/3777/3.jpg",
        3072,
        1728,
        1076196,
        "ac3f1512dfaa5d1f6c9ecfcbdbd536b843be7e4f3bba1151205f4fd767facf9a",
        "0.228009,0.000000,0.927083,1.000000",
        "nutrition-table",
    ),
    (
        "0037466016450_2",
        "0037466016450",
        "003/746/601/6450/2.jpg",
        1125,
        2000,
        429456,
        "a5dc650ea2c1925d8765f71725df3c65fa32c86ca39af38e0232ac39228b1208",
        "0.432500,0.227556,0.634000,0.447111",
        "nutrition-table",
    ),
    (
        "3068320112893_9",
        "3068320112893",
        "306/832/011/2893/9.jpg",
        1749,
        1200,
        531419,
        "ec27e12db38d3180a9722d8794d826e2b4dac35fae8aba18e6f1763ab0d8b455",
        "0.002500,0.011435,0.993374,1.000000",
        "nutrition-table",
    ),
    (
        "93300292_2",
        "93300292",
        "93300292/2.jpg",
        2000,
        1125,
        220769,
        "974475b8458b485d066586b5739d6d087e75aa242eba2dbdc6a7147c1a25f4bf",
        "0.526430,0.457050,0.967308,0.690774;0.129008,0.132886,0.307979,0.260403",
        "nutrition-table;nutrition-table-small-energy",
    ),
    (
        "2000000033325_2",
        "2000000033325",
        "200/000/003/3325/2.jpg",
        1024,
        133,
        68474,
        "c6638d0ee36e3a91f671f6b37b59a1f110aa33b8f61119f8d8dbfe4968ff973d",
        "0.060150,0.005859,0.977444,0.978516",
        "nutrition-table-text",
    ),
    (
        "00854252_6",
        "00854252",
        "00854252/6.jpg",
        480,
        640,
        67778,
        "7259224bccbf5b626bc459fc6d9b91e15a0682782e4cfe0d12ef91344b99c8b9",
        "0.321875,0.214583,0.598437,0.714583",
        "nutrition-table",
    ),
    (
        "01642582_1",
        "01642582",
        "01642582/1.jpg",
        1500,
        2000,
        703939,
        "cb857fcb0500db6e36d78cd7f4ffa0749dfd6a82c805c86ba16f5bb3375897b4",
        "0.623000,0.087333,0.759000,0.531333",
        "nutrition-table-text",
    ),
    (
        "0014100074120_2",
        "0014100074120",
        "001/410/007/4120/2.jpg",
        3120,
        4208,
        2114474,
        "915bc255945939f4a49a4dbd414d3795c9016a0edbb7ee45c7f2d4d74eca1d20",
        "0.222671,0.348077,0.535646,0.683654",
        "nutrition-table",
    ),
    (
        "0038000316104_4",
        "0038000316104",
        "003/800/031/6104/4.jpg",
        3024,
        4032,
        3926416,
        "0075618e7c4e65ad4c5c596778ff96dca24c6c4a343c1711ae5f50e32a2c0740",
        "0.063244,0.076389,0.565972,0.557209",
        "nutrition-table",
    ),
    (
        "0021000419074_1",
        "0021000419074",
        "002/100/041/9074/1.jpg",
        3120,
        4208,
        2426951,
        "fc57c3cf7c80bcb0b61137d88d9f4d5e61d3265043bedc83185014088de29645",
        "0.238408,0.136285,0.457484,0.485617",
        "nutrition-table",
    ),
    (
        "3250390768296_3",
        "3250390768296",
        "325/039/076/8296/3.jpg",
        2000,
        1500,
        930040,
        "b129570ece1aea447cf4b7e102462f742e373a618074749c83213f13a411f71d",
        "0.276667,0.113500,0.858000,0.922500",
        "nutrition-table",
    ),
    (
        "20298302_2",
        "20298302",
        "20298302/2.jpg",
        2000,
        2666,
        695634,
        "ffecf25f36b6e1227c8c815644864e615227508bf119e696650d4c2385cb23ca",
        "0.067142,0.554000,0.633964,0.935000",
        "nutrition-table",
    ),
    (
        "3284230002240_2",
        "3284230002240",
        "328/423/000/2240/2.jpg",
        2000,
        1500,
        1046374,
        "e1cea36548a06a73ce77447987962540fb111d9559621ae6f8fb1c341e46cba2",
        "0.248667,0.323000,0.862667,0.672000",
        "nutrition-table",
    ),
    (
        "26155432_2",
        "26155432",
        "26155432/2.jpg",
        1125,
        2000,
        674817,
        "796872eb60b0ee496a7880b8bb4f00aac0247ac1c5a968315091e58d7a67ba38",
        "0.191391,0.038288,0.877572,0.979910",
        "nutrition-table",
    ),
    (
        "3350031653285_1",
        "3350031653285",
        "335/003/165/3285/1.jpg",
        2000,
        2697,
        363191,
        "a03bccc8ffbf39c3bfd88f77597a63ebe7c624c93b5f6d5c3373b07f0bc434a3",
        "0.572103,0.515876,0.869470,0.778876",
        "nutrition-table",
    ),
    (
        "0072869110138_3",
        "0072869110138",
        "007/286/911/0138/3.jpg",
        2000,
        2666,
        418117,
        "c20222c10e515aa3153f3038fc4acf04f0c2d16c360dd11cf134125334f3a4c1",
        "0.087188,0.168500,0.840998,0.864143",
        "nutrition-table",
    ),
    (
        "20520090_2",
        "20520090",
        "20520090/2.jpg",
        2448,
        3264,
        666912,
        "61b384f0662e1427178d3b7f767ed66f6f488b443e197d67f43f5bd1e953eb12",
        "0.353248,0.052696,0.601716,0.867239",
        "nutrition-table-text",
    ),
    (
        "20608668_3",
        "20608668",
        "20608668/3.jpg",
        2448,
        3264,
        1082691,
        "1820191bd4a4583a4516c1c27e09cb252f5b3c67cd806946be3fa5b732578218",
        "0.212916,0.001759,0.694999,0.899510",
        "nutrition-table",
    ),
    (
        "0041498000028_3",
        "0041498000028",
        "004/149/800/0028/3.jpg",
        2000,
        3555,
        602639,
        "22119a6c288b2596ae2b7300a3fa00d059fba810f707352e3704aebfac78f1c0",
        "0.187342,0.310500,0.608102,0.722916",
        "nutrition-table",
    ),
    (
        "0052159000073_2",
        "0052159000073",
        "005/215/900/0073/2.jpg",
        3264,
        2448,
        994156,
        "06c7a86d3951c1bb0fc2d3d2e24945588ed0c98095ed1e30e47cd89a589f7e9f",
        "0.157449,0.379447,0.659068,0.978778",
        "nutrition-table",
    ),
    (
        "0063667090067_4",
        "0063667090067",
        "006/366/709/0067/4.jpg",
        648,
        2000,
        602529,
        "98f4fadb0cb00d40f23eb86a8dd3fcc37d3085ebeb6365431a2ebae125e32073",
        "0.178000,0.123457,0.615500,0.899691",
        "nutrition-table",
    ),
    (
        "0021130079278_2",
        "0021130079278",
        "002/113/007/9278/2.jpg",
        3120,
        4208,
        1662367,
        "87c9799c814a59b0ec4b9b31a963c5ca9f6fa3fa0354a1c8597f4888bbcdf3f6",
        "0.537785,0.263462,0.729567,0.700962",
        "nutrition-table",
    ),
    (
        "26101989_3",
        "26101989",
        "26101989/3.jpg",
        1812,
        1640,
        1317228,
        "2031c183e50ba3b51a473da441d5e6fe2f7108325223695e745a00f414087234",
        "0.013438,0.017673,0.968876,1.000000",
        "nutrition-table",
    ),
    (
        "0016000106406_2",
        "0016000106406",
        "001/600/010/6406/2.jpg",
        3120,
        4208,
        1981516,
        "b3ec29eb53c70583d37f2fd4b6db9e3a39f2f79170c57e46289b071bd4340a45",
        "0.000000,0.338141,0.275190,0.729487",
        "nutrition-table",
    ),
    (
        "20472313_2",
        "20472313",
        "20472313/2.jpg",
        1936,
        2592,
        823642,
        "ec05e9e351b2e6dc05965cb1155343b7cce8113f058d534f2228fb9d4acf6247",
        "0.371528,0.196475,0.606531,0.718621",
        "nutrition-table",
    ),
    (
        "0016000442825_4",
        "0016000442825",
        "001/600/044/2825/4.jpg",
        3024,
        1115,
        901714,
        "ceedef64f0322591aeb0869c05c479ab777a24acf57b0c5b0169058e9afbe45e",
        "0.160538,0.016534,0.993722,0.874669",
        "nutrition-table-text",
    ),
    (
        "20889869_2",
        "20889869",
        "20889869/2.jpg",
        2448,
        3264,
        1491430,
        "71c204a69e822c5ff784d640667e2cbd058d76f88b576aead4a93725579d93e5",
        "0.137788,0.034971,0.732202,0.452222",
        "nutrition-table",
    ),
    (
        "26212630_6",
        "26212630",
        "26212630/6.jpg",
        1315,
        964,
        391990,
        "8df94f947013552e95570c4ddd3a313d5f68149160d7d2ce841ca555f8e9b099",
        "0.006224,0.000000,0.783195,0.990875",
        "nutrition-table",
    ),
    (
        "3596710458455_7",
        "3596710458455",
        "359/671/045/8455/7.jpg",
        3024,
        4032,
        1775839,
        "5578487481bfcca0874d12348b9438b173c4e71680cf3f1cf44fa8d99069f692",
        "0.620784,0.243056,0.849950,0.576720",
        "nutrition-table",
    ),
    (
        "22000279_5",
        "22000279",
        "22000279/5.jpg",
        997,
        889,
        373764,
        "9af343b85dfc0ee30d9c906a36be307a0014f26d8adf166b1f1413c8dea8e619",
        "0.000000,0.003009,1.000000,1.000000",
        "nutrition-table",
    ),
    (
        "8480000342096_3",
        "8480000342096",
        "848/000/034/2096/3.jpg",
        1017,
        1262,
        372806,
        "ac52aea0714b4d35426ee365e3b985a15410bec7cb81e799b5c7fe9e92285e20",
        "0.343899,0.273353,0.690174,0.791544",
        "nutrition-table",
    ),
    (
        "0030000059708_1",
        "0030000059708",
        "003/000/005/9708/1.jpg",
        2988,
        5312,
        3484255,
        "c22514a0c6fd24641865ed9784c0ec969db7ff301939263584a9fb206c820cab",
        "0.338291,0.130857,0.473645,0.851406",
        "nutrition-table-text",
    ),
    (
        "3257984581972_1",
        "3257984581972",
        "325/798/458/1972/1.jpg",
        1329,
        1595,
        894087,
        "6854d18a35115bfd954b99020482957e02b6d00b8adc2282138c93b0f4df9bc9",
        "0.356306,0.093303,0.600627,0.471031",
        "nutrition-table",
    ),
    (
        "0070177067731_2",
        "0070177067731",
        "007/017/706/7731/2.jpg",
        2000,
        3536,
        461087,
        "399eaab42ce9718221f5be530f6c29a7235201f8ee6fd4e489f3af698912f7d2",
        "0.688348,0.405000,0.924208,0.991000",
        "nutrition-table",
    ),
    (
        "0058449771807_2",
        "0058449771807",
        "005/844/977/1807/2.jpg",
        816,
        1360,
        260911,
        "16e8ef416bcecd1eef48e2e7a1ae1da5b527da044bb4be78a47b5a857f49679a",
        "0.025648,0.071078,0.615457,0.895833",
        "nutrition-table",
    ),
    (
        "9300601462889_2",
        "9300601462889",
        "930/060/146/2889/2.jpg",
        2000,
        1125,
        316753,
        "7ea27af4ccc1edddc6dd3576b836c952fe02b4ab063b2049999bfcf9f2bcf209",
        "0.053333,0.053000,0.941447,0.309000",
        "nutrition-table",
    ),
    (
        "0073141152327_2",
        "0073141152327",
        "007/314/115/2327/2.jpg",
        3264,
        2448,
        1977998,
        "82d2bdcd0ab607512b3a94dd9f38fd33b3f2bcc433955d44115dabfed9e0bfeb",
        "0.221578,0.255145,0.536573,0.674174",
        "nutrition-table",
    ),
    (
        "0071962226104_3",
        "0071962226104",
        "007/196/222/6104/3.jpg",
        2000,
        1500,
        343182,
        "f20050293a58040677ba7587cd5872c4021a2b414870462a9e005d25a15a1018",
        "0.343414,0.176378,0.846631,0.719249",
        "nutrition-table",
    ),
    (
        "0039000081047_4",
        "0039000081047",
        "003/900/008/1047/4.jpg",
        2448,
        3264,
        1691932,
        "c0403fb05ad062a7e8e72026ef77b6c170f5048d536ad7aad0075e539b44fddb",
        "0.252638,0.218901,0.648284,0.770815",
        "nutrition-table",
    ),
    (
        "26209142_6",
        "26209142",
        "26209142/6.jpg",
        963,
        707,
        239573,
        "538876b40d562a5de8e7fcdc73efb2c422ef596dbd10d9d38614f55940c6a1ec",
        "0.000520,0.010788,1.000000,0.906074",
        "nutrition-table",
    ),
    (
        "20511586_3",
        "20511586",
        "20511586/3.jpg",
        1936,
        2592,
        843512,
        "4835e7dbda57c9f8f50964249292a0f64b79a1027366424f9500618c208b35da",
        "0.004195,0.220041,0.921162,0.621320",
        "nutrition-table",
    ),
    (
        "0071921377601_1",
        "0071921377601",
        "007/192/137/7601/1.jpg",
        2448,
        3264,
        1442656,
        "b72e33742f9b0a5b3904a0ed0f625d8ad19e1594abd6a07632bf806430bc9918",
        "0.149816,0.087010,0.530623,0.362496",
        "nutrition-table",
    ),
)

DEFAULT_CACHE_DIR = Path("weights") / "off-nutrition"  # working-directory-relative, like the notebook
SAMPLE_SEED = 42
SAMPLE_SPLIT = {
    "train": 71,
    "validation": 24,
    "test": 24,
}  # products (119 in the corpus; two have two photographs)
MIN_RECORDS = 8
MAX_RECORDS = 2_000
MAX_BOXES = MAX_DETECTIONS  # the decoder emits at most this many boxes, so a page cannot carry more targets
MIN_BOX_SIDE = 4.0  # pixels
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def image_url(path: str) -> str:
    return f"{CORPUS_BASE_URL}{path}"


def fetch_corpus(*, cache_dir: str | Path | None = None, fetcher: Any = None) -> dict[str, bytes]:
    """Return every pinned photograph (bytes keyed by record id) from the cache or the Open Food Facts host.

    Every file is refused on a byte-size or SHA-256 mismatch against `SAMPLE_RECORDS`.
    """
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for rid, _barcode, path, _w, _h, size, digest, _boxes, _cats in SAMPLE_RECORDS:
        local = cache / f"{rid}.jpg"
        data = local.read_bytes() if local.is_file() else b""
        if len(data) != size or _sha256_bytes(data) != digest:
            url = image_url(path)
            if fetcher is not None:
                data = fetcher(url)
            else:
                request = urllib.request.Request(
                    url, headers={"User-Agent": "dimer-table-transformer-tutorial/1.0"}
                )
                with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310 (pinned https URL)
                    data = response.read()
            if len(data) != size or _sha256_bytes(data) != digest:
                raise ValueError(
                    f"{rid} ({path}): fetched {len(data)} bytes with sha256 {_sha256_bytes(data)[:16]}…, "
                    f"pinned {size} / {digest[:16]}…"
                )
            local.write_bytes(data)
        out[rid] = data
    return out


def read_corpus(files: Mapping[str, bytes]) -> list[dict[str, Any]]:
    """Decode the verified photographs into `{id, image, boxes}` records (pixel xyxy) with provenance."""
    out = []
    for rid, barcode, path, width, height, _size, _digest, boxes, cats in SAMPLE_RECORDS:
        if rid not in files:
            raise ValueError(f"corpus is missing {rid}")
        image = Image.open(io.BytesIO(files[rid]))
        image.load()
        if image.size != (width, height):
            raise ValueError(f"{rid}: served image is {image.size}, the labelled image was {(width, height)}")
        image = image.convert("RGB")
        scale = min(1.0, CORPUS_LONGEST_SIDE / max(width, height))
        if scale < 1.0:
            image = image.resize((round(width * scale), round(height * scale)), Image.LANCZOS)
        pixel_boxes = []
        kept_categories = []
        dropped = 0
        for chunk, category in zip(boxes.split(";"), cats.split(";"), strict=True):
            ymin, xmin, ymax, xmax = (float(v) for v in chunk.split(","))
            box = [xmin * image.width, ymin * image.height, xmax * image.width, ymax * image.height]
            if box[2] - box[0] < MIN_BOX_SIDE or box[3] - box[1] < MIN_BOX_SIDE:
                dropped += 1  # one annotation in the corpus is a 1.5 x 4 px sliver; the contract refuses it
                continue
            pixel_boxes.append(box)
            kept_categories.append(category)
        out.append(
            {
                "id": rid,
                "image": image,
                "boxes": pixel_boxes,
                "source_size": [width, height],
                "dropped_boxes": dropped,
                "categories": kept_categories,
                "barcode": barcode,
                "image_url": image_url(path),
            }
        )
    return out


def build_sample_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded draw of whole products (by barcode) into train / validation / test: `sizes` counts products."""
    sizes = dict(sizes or SAMPLE_SPLIT)
    rng = random.Random(seed)
    by_product: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_product.setdefault(_split_unit(record), []).append(dict(record))
    products = sorted(by_product)
    rng.shuffle(products)
    needed = sum(sizes.values())
    if len(products) < needed:
        raise ValueError(f"only {len(products)} products available, need {needed}")
    out: dict[str, list[dict[str, Any]]] = {}
    cursor = 0
    for name, count in sizes.items():
        part = [r for product in products[cursor : cursor + count] for r in by_product[product]]
        cursor += count
        rng.shuffle(part)
        out[name] = [{**r, "id": f"{name}-{i:03d}", "source_id": r["id"]} for i, r in enumerate(part)]
    return out


def fetch_sample_dataset(
    *,
    cache_dir: str | Path | None = None,
    fetcher: Any = None,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """The tutorial splits from the pinned corpus."""
    return build_sample_dataset(
        read_corpus(fetch_corpus(cache_dir=cache_dir, fetcher=fetcher)), seed=seed, sizes=sizes
    )


def _check_record(record: Any, index: int) -> dict[str, Any]:
    label_name = f"records[{index}]"
    if not isinstance(record, Mapping):
        raise ValueError(f"{label_name} must be a mapping with id/image/boxes")
    for key in ("id", "image", "boxes"):
        if key not in record:
            raise ValueError(f"{label_name} is missing {key!r}")
    rid, image, boxes = record["id"], record["image"], record["boxes"]
    if not isinstance(rid, str) or not _ID_RE.match(rid):
        raise ValueError(f"{label_name}: id must match {_ID_RE.pattern}")
    if isinstance(image, str | Path):
        path = Path(image)
        if not path.is_file():
            raise ValueError(f"{label_name}: image file not found: {path}")
        image = Image.open(path)
        image.load()
    if not isinstance(image, Image.Image):
        raise ValueError(f"{label_name}: image must be a PIL.Image.Image or a file path")
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIDE or max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(
            f"{label_name}: image side outside {MIN_IMAGE_SIDE}..MAX_IMAGE_SIDE={MAX_IMAGE_SIDE} px: "
            f"{image.size}"
        )
    if isinstance(boxes, str | bytes) or not isinstance(boxes, Sequence) or not 1 <= len(boxes) <= MAX_BOXES:
        raise ValueError(
            f"{label_name}: boxes must be a list of 1..{MAX_BOXES} [x_min, y_min, x_max, y_max] boxes"
        )
    checked_boxes = []
    for b, box in enumerate(boxes):
        if isinstance(box, str | bytes) or not isinstance(box, Sequence) or len(box) != 4:
            raise ValueError(f"{label_name}: boxes[{b}] must have four values")
        x0, y0, x1, y1 = (float(v) for v in box)
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(
                f"{label_name}: boxes[{b}] {[x0, y0, x1, y1]} must lie inside the {image.size} image "
                "with x0 < x1 and y0 < y1"
            )
        if x1 - x0 < MIN_BOX_SIDE or y1 - y0 < MIN_BOX_SIDE:
            raise ValueError(f"{label_name}: boxes[{b}] is smaller than {MIN_BOX_SIDE} px on a side")
        checked_boxes.append([x0, y0, x1, y1])
    item = {"id": rid, "image": image.convert("RGB"), "boxes": checked_boxes}
    for key in ("source_id", "categories", "barcode", "image_url", "group", "source_size", "dropped_boxes"):
        if key in record:
            item[key] = record[key]
    return item


def validate_dataset(
    records: Sequence[Mapping[str, Any]], *, min_records: int = MIN_RECORDS, max_records: int = MAX_RECORDS
) -> dict[str, Any]:
    """Structural validation of a box-labelled image dataset; raises ValueError before any model import."""
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or isinstance(records, str | bytes):
        raise ValueError("records must be a list of {id, image, boxes} mappings")
    if not min_records <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; {min_records}..{max_records} are required")
    checked = [_check_record(record, index) for index, record in enumerate(records)]
    ids = [r["id"] for r in checked]
    if len(set(ids)) != len(ids):
        duplicate = next(i for i in ids if ids.count(i) > 1)
        raise ValueError(f"duplicate id {duplicate!r}")
    sides = [max(r["image"].size) for r in checked]
    n_boxes = [len(r["boxes"]) for r in checked]
    fractions = [
        (b[2] - b[0]) * (b[3] - b[1]) / (r["image"].size[0] * r["image"].size[1])
        for r in checked
        for b in r["boxes"]
    ]
    return {
        "records": checked,
        "n_records": len(checked),
        "n_boxes": sum(n_boxes),
        "boxes_per_image": {"min": min(n_boxes), "max": max(n_boxes)},
        "image_side": {"min": min(sides), "max": max(sides)},
        "box_area_fraction": {"min": round(min(fractions), 4), "max": round(max(fractions), 4)},
        "digest": dataset_digest(checked),
        "model_id": MODEL_ID,
    }


def image_digest(image: Image.Image) -> str:
    """SHA-256 of the decoded RGB pixels (size + bytes), so a re-encoded copy of the same photo matches."""
    rgb = image.convert("RGB")
    return _sha256_bytes(f"{rgb.size[0]}x{rgb.size[1]}:".encode() + rgb.tobytes())


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        [r["id"], image_digest(r["image"]), [[round(float(v), 2) for v in b] for b in r["boxes"]]]
        for r in records
    ]
    return _sha256_bytes(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _split_unit(record: Mapping[str, Any]) -> str:
    return str(record.get("group") or record.get("barcode") or record.get("source_id") or record["id"])


def check_split_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Assert no photograph (by decoded-pixel digest) and no product appears in two splits (leakage check)."""
    seen: dict[str, str] = {}
    units: dict[str, str] = {}
    for name, records in splits.items():
        for record in records:
            key = image_digest(record["image"])
            if key in seen and seen[key] != name:
                raise ValueError(f"image {record['id']!r} appears in both {seen[key]} and {name}")
            seen[key] = name
            unit = _split_unit(record)
            if unit in units and units[unit] != name:
                raise ValueError(f"product {unit!r} has photographs in both {units[unit]} and {name}")
            units[unit] = name
    return {name: len(records) for name, records in splits.items()}


def split_summary(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Photographs, boxes and products per split (an observation of what the split unit was)."""
    return {
        name: {
            "images": len(records),
            "boxes": sum(len(r["boxes"]) for r in records),
            "products": len({_split_unit(r) for r in records}),
        }
        for name, records in splits.items()
    }


def split_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded shuffle of a BYOD dataset into train/validation/test by split unit (`group` / `barcode`, else
    the image itself) after de-duplicating photographs."""
    if not (0.0 <= val_fraction < 1.0 and 0.0 < test_fraction < 1.0 and val_fraction + test_fraction < 1.0):
        raise ValueError("fractions must satisfy 0 <= val < 1, 0 < test < 1, val + test < 1")
    checked = validate_dataset(records)["records"]
    seen: set[str] = set()
    by_unit: dict[str, list[dict[str, Any]]] = {}
    for record in checked:
        key = image_digest(record["image"])
        if key not in seen:
            seen.add(key)
            by_unit.setdefault(_split_unit(record), []).append(record)
    rng = random.Random(seed)
    units = sorted(by_unit)
    rng.shuffle(units)
    n_test = max(1, round(len(units) * test_fraction))
    n_val = round(len(units) * val_fraction)
    splits: dict[str, list[dict[str, Any]]] = {"test": [], "validation": [], "train": []}
    for name, chosen in (
        ("test", units[:n_test]),
        ("validation", units[n_test : n_test + n_val]),
        ("train", units[n_test + n_val :]),
    ):
        for unit in chosen:
            splits[name].extend(by_unit[unit])
    for part in splits.values():
        rng.shuffle(part)
    if len(splits["train"]) < MIN_RECORDS:
        raise ValueError(
            f"split leaves {len(splits['train'])} training records; at least {MIN_RECORDS} are required"
        )
    return splits


def load_byod_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Read `{id, image, boxes}` records from a directory or a zip holding `boxes.csv` (columns `id`, `file`,
    `x_min`, `y_min`, `x_max`, `y_max`, optional `group`; one row per box, pixel coordinates) beside the image
    files; images are decoded, never extracted to disk."""
    source = Path(path)
    if source.is_dir():
        table = (source / "boxes.csv").read_text(encoding="utf-8")
        loader = lambda name: Image.open(source / name)  # noqa: E731
    elif source.is_file() and source.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(source)
        members = {Path(n).name: n for n in archive.namelist()}
        if "boxes.csv" not in members:
            raise ValueError("BYOD zip must contain boxes.csv")
        table = archive.read(members["boxes.csv"]).decode("utf-8")
        loader = lambda name: Image.open(io.BytesIO(archive.read(members[name])))  # noqa: E731
    else:
        raise ValueError("BYOD datasets must be a directory or a .zip holding boxes.csv and the image files")
    rows = list(csv.DictReader(io.StringIO(table)))
    missing = {"id", "file", "x_min", "y_min", "x_max", "y_max"} - set(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"boxes.csv is missing columns {sorted(missing)}")
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        item = grouped.get(row["id"])
        if item is None:
            image = loader(row["file"])
            image.load()
            item = {"id": row["id"], "image": image.convert("RGB"), "boxes": []}
            if row.get("group"):
                item["group"] = row["group"]
            grouped[row["id"]] = item
        item["boxes"].append(
            [float(row["x_min"]), float(row["y_min"]), float(row["x_max"]), float(row["y_max"])]
        )
    return list(grouped.values())


def write_dataset_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write the boxes table of a split (one row per box, provenance) in the shape BYOD expects."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "file", "x_min", "y_min", "x_max", "y_max", "group", "category", "image_url"],
        )
        writer.writeheader()
        for record in records:
            cats = record.get("categories") or [CLASS_NAME] * len(record["boxes"])
            for box, cat in zip(record["boxes"], cats, strict=True):
                writer.writerow(
                    {
                        "id": record["id"],
                        "file": f"{record.get('source_id') or record['id']}.jpg",
                        "x_min": round(box[0], 2),
                        "y_min": round(box[1], 2),
                        "x_max": round(box[2], 2),
                        "y_max": round(box[3], 2),
                        "group": _split_unit(record),
                        "category": cat,
                        "image_url": record.get("image_url", ""),
                    }
                )
    return out
