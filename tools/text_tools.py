import re
import string
import unicodedata

def normalize_diacritics(source, new_style=False, decomposed=False):
    tone = r'([\u0300\u0309\u0303\u0301\u0323])'
    combining_breve = r'\u0306'
    combining_circumflex_accent = r'\u0302'
    combining_horn = r'\u031B'
    diacritics = f'{combining_breve}{combining_circumflex_accent}{combining_horn}'
    result = unicodedata.normalize('NFD', source)
    # Put the tone on the second vowel
    result = re.sub(r'(?i){}([aeiouy{}]+)'.format(tone, diacritics), r'\2\1', result)
    # Put the tone on the vowel with a diacritic
    result = re.sub(r'(?i)(?<=[{}])(.){}'.format(diacritics, tone), r'\2\1', result)
    # For vowels that are not oa, oe, uy put the tone on the penultimate vowel
    result = re.sub(r'(?i)(?<=[ae])([iouy]){}'.format(tone), r'\2\1', result)
    result = re.sub(r'(?i)(?<=[oy])([iuy]){}'.format(tone), r'\2\1', result)
    result = re.sub(r'(?i)(?<!q)(u)([aeiou]){}'.format(tone), r'\1\3\2', result)
    result = re.sub(r'(?i)(?<!g)(i)([aeiouy]){}'.format(tone), r'\1\3\2', result)
    if not new_style:
        # Put tone in the symmetrical position
        result = re.sub(r'(?i)(?<!q)([ou])([aeoy]){}(?!\w)'.format(tone), r'\1\3\2', result)
    if decomposed:
        return unicodedata.normalize('NFD', result)
    return unicodedata.normalize('NFC', result)


# Taken from CLDR - Unicode Common Locale Data Repository
# http://demo.icu-project.org/icu-bin/locexp
ACCENTED_CHARACTERS = [
    'a',
    'à',
    'ả',
    'ã',
    'á',
    'ạ',
    'ă',
    'ằ',
    'ẳ',
    'ẵ',
    'ắ',
    'ặ',
    'â',
    'ầ',
    'ẩ',
    'ẫ',
    'ấ',
    'ậ',
    'b',
    'c',
    'd',
    'đ',
    'e',
    'è',
    'ẻ',
    'ẽ',
    'é',
    'ẹ',
    'ê',
    'ề',
    'ể',
    'ễ',
    'ế',
    'ệ',
    'f',
    'g',
    'h',
    'i',
    'ì',
    'ỉ',
    'ĩ',
    'í',
    'ị',
    'j',
    'k',
    'l',
    'm',
    'n',
    'o',
    'ò',
    'ỏ',
    'õ',
    'ó',
    'ọ',
    'ô',
    'ồ',
    'ổ',
    'ỗ',
    'ố',
    'ộ',
    'ơ',
    'ờ',
    'ở',
    'ỡ',
    'ớ',
    'ợ',
    'p',
    'q',
    'r',
    's',
    't',
    'u',
    'ù',
    'ủ',
    'ũ',
    'ú',
    'ụ',
    'ư',
    'ừ',
    'ử',
    'ữ',
    'ứ',
    'ự',
    'v',
    'w',
    'x',
    'y',
    'ỳ',
    'ỷ',
    'ỹ',
    'ý',
    'ỵ',
    'z',
]
VIETNAMESE_CHARACTERS = list(string.punctuation) + list(string.digits) + [x.upper() for x in ACCENTED_CHARACTERS] + ACCENTED_CHARACTERS


class VietnameseOrderDict(dict):
    def __init__(self):
        for i, char in enumerate(VIETNAMESE_CHARACTERS):
            self[char] = i

    def __getitem__(self, k):
        # Any characters that is not in the dictinary has its index shifted out by the number of keys in the dictionary
        # to preserve relative order and give precedence to characters used in Vietnamese
        if k not in self:
            return ord(k) + len(self)
        return super().__getitem__(k)


vietnamese_order_dict = VietnameseOrderDict()


def vietnamese_sort_key(word):
    word = unicodedata.normalize('NFC', word)
    return [vietnamese_order_dict[c] for c in word]


def vietnamese_case_insensitive_sort_key(word):
    word = unicodedata.normalize('NFC', word)
    return [vietnamese_order_dict[c] for c in word.lower()]

# pattern = "áàảãạắằẳẵặấầẩẫậèéẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
# replacement = "a"*5 + "ă"*5 + "â"*5 + "e"*5 + "ê"*5 + "i"*5 + "o"*5 + "ô"*5 + "ơ"*5 + "u"*5 + "ư"*5 + "y"*5

pattern = "ạặậ" + "ẹệ" + "ị" + "ọộợ" + "ụự" + "ỵ" + "ếềèé"
replacement = "aăâ" + "eê" + "i" + "oôơ" + "uư" + "y" + "ê"*4

BANG_XOA_DAU = str.maketrans(
    pattern,
    replacement,
)

def delete_diacritic(txt: str) -> str:
    if not unicodedata.is_normalized("NFC", txt):
        txt = unicodedata.normalize("NFC", txt)
    return txt.translate(BANG_XOA_DAU)


accent_dictionary = "aàáạảãâầấậẩẫăằắặẳẵAÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪeèéẹẻẽêềếệểễEÈÉẸẺẼÊỀẾỆỂỄoòóọỏõôồốộổỗơờớợởỡOÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠiìíịỉĩIÌÍỊỈĨuùúụủũưừứựửữƯỪỨỰỬỮUÙÚỤỦŨyỳýỵỷỹYỲÝỴỶỸ"

def make_groups():
    groups = []
    i = 0
    while i < len(accent_dictionary) - 5:
        group = [c for c in accent_dictionary[i : i + 6]]
        i += 6
        groups.append(group)
    return groups


groups = make_groups()

TONES = ["", "1", "2", "3", "4", "5"]
SOURCES = ["ă", "â", "Ă", "Â", "ê", "Ê", "ô", "ơ", "Ô", "Ơ", "ư", "Ư", "Đ", "đ"]
TARGETS = ["a6", "a0", "A6", "A0", "e0", "E0", "o0", "o7", "O0", "O7", "u8", "U8", "D9", "d9"]

def parse_tone(word):
    res = ""
    for char in word:
        if char in accent_dictionary:
            for group in groups:
                if char in group:
                    res += group[0]
                    tone = TONES[group.index(char)]
                    res += tone
        else:
            res += char
    return res


def tone_encode(word):
    word = parse_tone(word)
    res = ""
    for char in word:
        if char in SOURCES:
            res += TARGETS[SOURCES.index(char)]
        else:
            res += char
    return res


def tone_decode(recognition):
    for char in TARGETS:
        recognition = recognition.replace(char, SOURCES[TARGETS.index(char)])
    res = []
    for i, char in enumerate(recognition):
        if char in TONES:
            if i == 0 or len(res) == 0:
                continue
            else:
                for group in groups:
                    if res[-1] == group[0]:
                        res[-1] = group[TONES.index(char)]
        elif char not in ["0", "6", "7", "8", "9"]:
            res.append(char)

    recognition = "".join(res)
    return recognition