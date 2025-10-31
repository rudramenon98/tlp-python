import itertools
import re
import statistics
import string
from collections import defaultdict

import fitz
import intervals as I
import numpy as np
import textdistance
import unidecode


##Remmove Prefix
def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text  # or whatever


##Line info
def line_info(block):
    """return info of a line including font_family and font_size, and coordinate of left most char in the line"""

    """the process is done only for the first detected font in the line"""

    for font in block:

        for char in font:

            font_family = font.attrib["name"]

            font_size = font.attrib["size"]

            most_left_x = block.attrib["bbox"].split(" ")[0]

            most_left_y = block.attrib["bbox"].split(" ")[1]

            x0 = char.attrib["x"]

            y0 = char.attrib["y"]

            return font_family, font_size, most_left_x, most_left_y, x0, y0

            break

        break


##accent handler
def accent_and_badtext_handler(badtext):

    try:
        encoded = str(badtext).encode("cp1252")
        goodtext = encoded.decode("utf-8")
    except:
        accented_string = str(badtext)
        badtext = unidecode.unidecode(accented_string)
        encoded = badtext.encode("cp1252")
        goodtext = encoded.decode("utf-8")

    return goodtext


def remove_everything_except_(text):
    one_letter = list(string.ascii_uppercase) + list(string.ascii_lowercase)
    for letter in text:
        if letter not in one_letter:  # +digit_letter:
            text = text.replace(letter, "")
    return text


def return_line_of_text(line):
    """return a line of text in PDF"""

    list_char = []
    for font in line:
        for char in font:
            list_char.append(char.attrib["c"])
    return "".join(list_char)


def line_prev_and_next(line):

    list_char = []
    for font in line:
        for char in font:
            list_char.append(char.attrib["c"])
    return "".join(list_char)


def check_order_of_texts(check_list, content):
    indices = []
    for _, ele in enumerate(check_list):
        indices.append(content.index(ele.strip()))

    if (
        "".join(check_list) in content
    ):  # and list(sorted(indices))==indices or textdistance.levenshtein("".join(check_list), content)<3) :
        return True
    else:
        return False


def check_available_words(check_list, content):
    flag = True
    for sent in check_list:
        for word in sent.split():
            if word not in content.split():
                flag = False
                return False
                break
        if flag == False:
            break
        if flag == True:
            if not check_order_of_texts(
                sent.split(), content[content.index(sent.strip()) :]
            ):
                flag = False
                return False
                break

    return True if flag == True else False


def remove_punc_(text_line):
    for punc in Symbols().all_punctuations:
        text_line = str(text_line).replace(punc, "")
    return text_line


def remove_white_space(text_line):
    return str(text_line).replace(" ", "")


def pre_processing_text_before_labeling(text_line):

    text_line = remove_punc_(text_line)

    text = remove_white_space(text_line)

    if text.startswith("§"):
        text = text[1:].strip()
        text_line = text_line[1:].strip()

    if "-" in text:
        if text[-1] == "-":
            text = text[:-1]
            text_line = text_line[:-1]

    if text[-2:] == ".." and text.count(".") > 2:
        text = text[: text.index("..")]
        try:
            text_line = text_line[: text_line.index("..")]
        except:
            text_line = text_line[: text_line.index(". .")]
    return text, text_line


def extract_raw_text(row):
    raw_string = ""
    flag = True
    for _, char in enumerate(str(row)):
        if char == "<":
            flag = False
            continue
        if char == ">":
            flag = True
            continue
        if flag == True:
            raw_string = raw_string + char
    return raw_string


def longest_common_string(text, content):

    start_index = None
    end_index = None
    score = None

    try:
        textdistance.lcsseq(text, content)
        longest_common_substring = textdistance.lcsstr(text, content)
        if content.count(longest_common_substring) > 1:
            start_index = content.index(longest_common_substring)
            end_index = start_index + len(longest_common_substring)
            content = "".join([content[:start_index], content[end_index:]])
        score = textdistance.ratcliff_obershelp(longest_common_substring, text)
        start_index = content.index(longest_common_substring)
        end_index = start_index + len(longest_common_substring)

    except:
        return start_index, end_index, score, longest_common_substring

    return start_index, end_index, score, longest_common_substring


def letter_incommon(text1, text2):
    text1 = text1.lower().strip()
    text2 = text2.lower().strip()
    c = 0
    if text1 and text2 and text1 != "" and text2 != "":
        for _ in range(0, min(len(text1), len(text2))):
            if text1[_] == text2[_]:
                c = c + 1
        if c >= min(len(text1), len(text2)) / 2:
            return True
        else:
            return False
    else:
        return False


# Pre defined lists:


class Symbols:

    def __init__(self):

        self.terms_assigned_for_arxiv_headers_inManualPdf = [
            "section"
        ]  # manual pdfs are pdfs that are created by Berckan manulally from a fixed latex file. sometimes in pdf they have extra term at the begining like "section"
        self.punctuations_in_arxiv_headers_inManualPdf = ["(", ")", "-"]

        self.punctuations = list(string.punctuation)

        self.roman_letters = {
            1: "I",
            2: "II",
            3: "III",
            4: "IV",
            5: "V",
            6: "VI",
            7: "VII",
            8: "VIII",
            9: "IX",
            10: "X",
            11: "XI",
            12: "XII",
            13: "XIII",
            14: "XIV",
            15: "XV",
            16: "XVI",
            17: "XVII",
            18: "XVIII",
            19: "XIX",
            20: "XX",
            21: "XXI",
            22: "XXII",
            23: "XXIII",
            24: "XXIV",
            30: "XXX",
            40: "XL",
            50: "L",
            60: "LX",
            70: "LXX",
            80: "LXXX",
            90: "XC",
            100: "C",
            101: "CI",
            102: "CII",
            200: "CC",
            300: "CCC",
            400: "CD",
            500: "D",
            600: "DC",
            700: "DCC",
            800: "DCCC",
            900: "CM",
            1000: "M",
            1001: "MI",
            1002: "MII",
            1003: "MIII",
            1900: "MCM",
            2000: "MM",
            2001: "MMI",
            2002: "MMII",
            2100: "MMC",
            3000: "MMM",
            4000: "MMMMor M V",
            5000: "V",
        }

        self.latex_symbol_letters = [
            " ",
            ",",
            ":",
            ";",
            "‐",
            "’",
            "'",
            "′",
            "″",
            "‴",
            ".",
            "&",
            "@",
            "^",
            "/",
            "\\",
            "…",
            "*",
            "⁂",
            "*  *  *",
            "-",
            "‒",
            "–",
            "—",
            "=",
            "⸗",
            "?",
            "!",
            "‽",
            "¡",
            "¿",
            "!",
            "?",
            "⸮",
            "№",
            "º",
            "ª",
            "%," "‰",
            "‱",
            "°",
            "⌀",
            "+",
            "−",
            "×",
            "÷",
            "~",
            "±",
            "∓",
            "–",
            "_",
            "⁀",
            "|",
            "¦",
            "‖",
            "•",
            "·",
            "©",
            "©",
            "℗",
            "®",
            "SM",
            "TM",
            "‘",
            "’",
            "“",
            "”",
            "'",
            "'",
            '"',
            '"',
            "‹",
            "›",
            "<",
            ">",
            "«",
            "»",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            "⟨",
            "⟩",
            "”",
            "〃",
            "†",
            "‡",
            "❧",
            "☞",
            "◊",
            "¶",
            "⸿",
            "፠",
            "๛",
            "※",
            "#",
            "§",
        ]

        self.latex_commands = [
            r"\rightarrow",
            r"\leftarrow",
            r"\\addcontentsline",
            r"\\addtocontents",
            r"\\addtocounter",
            r"\\address",
            r"\\addtolength",
            r"\\addvspace",
            r"\\alpha",
            r"\\alph",
            r"\\appendix",
            r"\\arabic",
            r"\\author",
            r"\\backslash",
            r"\\baselineskip",
            r"\\baselinestretch",
            r"\\begin",
            r"\\beta",
            r"\\bf",
            r"\\bibitem",
            r"\\bigskip",
            r"\\boldmath",
            r"\cal",
            r"\caption",
            r"\cdots",
            r"\centering",
            r"\circle",
            r"\cite",
            r"\cleardoublepage",
            r"\clearpage",
            r"\cline",
            r"\closing",
            r"\dashbox",
            r"\date",
            r"\ddots",
            r"\dfrac",
            r"\dotfill",
            r"\em",
            r"\end",
            r"\ensuremath",
            r"\\fbox",
            r"\\flushbottom",
            r"\\fnsymbol",
            r"\\footnote",
            r"\\footnotemark",
            r"\\footnotesize",
            r"\\footnotetext",
            r"\\frac",
            r"\\frame",
            r"\\framebox",
            r"\\frenchspacing",
            r"\gamma",
            r"\hfill",
            r"\hline",
            r"\hrulefill",
            r"\hspace",
            r"\huge",
            r"\Huge",
            r"\hyphenation",
            r"\iint",
            r"\include",
            r"\includeonly",
            r"\indent",
            r"\infty",
            r"\input",
            r"\int",
            r"\it",
            r"\item",
            r"\kill",
            r"\label",
            r"\large",
            r"\Large",
            r"\LARGE",
            r"\left",
            r"\ldots",
            r"\left",
            r"\lefteqn",
            r"\line",
            r"\linebreak",
            r"\linethickness",
            r"\linewidth",
            r"\location",
            r"\makebox",
            r"\maketitle",
            r"\markboth",
            r"\markright",
            r"\mathcal",
            r"\mathop",
            r"\mbox",
            r"\medskip",
            r"\multicolumn",
            r"\\multiput",
            r"\\newcommand",
            r"\\newcounter",
            r"\\newenvironment",
            r"\\newfont",
            r"\\newlength",
            r"\\newline",
            r"\\newpage",
            r"\\newsavebox",
            r"\\newtheorem",
            r"\\nocite",
            r"\\noindent",
            r"\\nolinebreak",
            r"\\normalsize ",
            r"\\nopagebreak",
            r"\\not",
            r"\oint",
            r"\onecolumn",
            r"\opening",
            r"\oval",
            r"\overbrace",
            r"\overline",
            r"\pagebreak",
            r"\pagenumbering",
            r"\pageref",
            r"\pagestyle",
            r"\par",
            r"\parbox",
            r"\parindent",
            r"\parskip",
            r"\protect",
            r"\providecommand",
            r"\put",
            r"\\raggedbottom",
            r"\\raggedleft",
            r"\\raggedright",
            r"\\raisebox",
            r"\\ref",
            r"\\renewcommand",
            r"\\right",
            r"\\rm",
            r"\\roman",
            r"\\rule",
            r"\savebox",
            r"\sbox",
            r"\sc",
            r"\scriptsize",
            r"\section",
            r"\setcounter",
            r"\setlength",
            r"\settowidth",
            r"\sf",
            r"\shortstack",
            r"\signature",
            r"\sl",
            r"\small",
            r"\smallskip",
            r"\sqrt",
            r"\stackrel",
            r"\subsection",
            r"\subsubsection",
            r"\sum",
            r"\\tableofcontents",
            r"\\telephone",
            r"\\textwidth",
            r"\\textheight",
            r"\\thanks",
            r"\\thispagestyle",
            r"\\tiny",
            r"\\title",
            r"\\today",
            r"\\tt",
            r"\\twocolumn",
            r"\\typeout",
            r"\\typein",
            r"\underbrace",
            r"\underline",
            r"\unitlength",
            r"\usebox",
            r"\usecounter",
            r"\value",
            r"\\vdots",
            r"\\vector",
            r"\\verb",
            r"\\vfill",
            r"\\vline",
            r"\\vphantom",
            r"\\vspace",
            r"\<space>",
            r"\paragraph",
        ]

        self.latex_math_commands = [
            r"\sin",
            r"\cos",
            r"\\tan",
            r"\cot",
            r"\\arccot",
            r"\\arctan",
            r"\\arccos",
            r"\\arcsin",
            r"\sinh",
            r"\cosh",
            r"\\tanh",
            r"\coth",
            r"\sec",
            r"\csc",
            r"\in",
            r"\partial",
            r"\imath",
            r"\Re",
            r"\\nabla",
            r"\\aleph",
            r"\eth",
            r"\jmath",
            r"\Im",
            r"\Box",
            r"\\beth",
            r"\gimel",
            r"\infty",
            r"\wp",
            r"\ell",
            r"\hbar",
            r"\omega",
            r"\Omega",
            r"\mu",
            r"\psi",
            r"\Psi",
            r"\lambda",
            r"\Lambda",
            r"\chi",
            r"\\varkappa",
            r"\kappa",
            r"\\varphi",
            r"\Phi",
            r"\phi",
            r"\iota",
            r"\\upsilon",
            r"\\vartheta",
            r"\Theta",
            r"\\theta",
            r"\\tau",
            r"\eta",
            r"\\varsigma",
            r"\sigma",
            r"\Sigma",
            r"\zeta",
            r"\\varrho",
            r"\\rho",
            r"\\varepsilon",
            r"\epsilon",
            r"\\varpi",
            r"\Pi",
            r"\pi",
            r"\delta",
            r"\Delta",
            r"\gamma",
            r"\Gamma",
            r"\Xi",
            r"\\xi",
            r"\\beta",
            r"\\nu",
            r"\\alpha",
            r"\\rfloor",
            r"\lfloor",
            r"\Downarrow",
            r"\downarrow",
            r"\\rceil",
            r"\lceil",
            r"\\Uparrow",
            r"\\uparrow",
            r"\\rangle",
            r"\langle",
            r"\\backslash",
            r"\\rightleftharpoons",
            r"\\varnothing",
            r"\emptyset",
            r"\\angle",
            r"\\bot",
            r"\lor",
            r"\\top",
            r"\Leftrightarrow",
            r"\\ni",
            r"\iff",
            r"\\notin",
            r"\leftrightarrow",
            r"\in",
            r"\implies",
            r"\Rightarrow",
            r"\supset",
            r"\impliedby",
            r"\subset",
            r"\cup",
            r"\cap",
            r"\implies",
            r"\\neg",
            r"\mapsto",
            r"\\forall",
            r"\gets",
            r"\leftarrow",
            r"\\nexists",
            r"\\to",
            r"\\rightarrow",
            r"\exists",
            r"\\amalg",
            r"\\amalg",
            r"\cdot",
            r"\ddagger",
            r"\setminus",
            r"\\bullet",
            r"\wedge",
            r"\dagger",
            r"\circ",
            r"\\bigcirc",
            r"\\vee",
            r"\star",
            r"\odot",
            r"\\triangleright",
            r"\sqcup",
            r"\\ast",
            r"\oslash",
            r"\\triangleleft",
            r"\sqcap",
            r"\div",
            r"\otimes",
            r"\\bigtriangledown",
            r"\\uplus",
            r"\\times",
            r"\ominus",
            r"\\bigtriangleup",
            r"\cup",
            r"\mp",
            r"\oplus",
            r"\diamond",
            r"\cap",
            r"\pm",
            r"\measuredangle",
            r"\sphericalangle",
            r"\\neq",
            r"\succeq",
            r"\preceq",
            r"\succv",
            r"\propto",
            r"\sqsupseteq",
            r"\sqsubseteq",
            r"\mid",
            r"\perp",
            r"\sim",
            r"\sqsupset",
            r"\sqsubset",
            r"\\notin",
            r"\models",
            r"\simeq",
            r"\\nsupseteq",
            r"\\nsubseteq",
            r"\\frown",
            r"\smile",
            r"\cong",
            r"\supseteq",
            r"\subseteq",
            r"\\ni",
            r"\in",
            r"\\approx",
            r"\supset",
            r"\subset",
            r"\dashv",
            r"\\vdash",
            r"\equiv",
            r"\gg",
            r"\ll",
            r"\\bowtie",
            r"\\asymp",
            r"\doteq",
            r"\geq",
            r"\leq",
            r"\\nparallel",
            r"\parallel",
            r"\hdotsfor",
            r"\iddots",
            r"\ddots",
            r"\\vdots",
            r"\cdots",
            r"\ldots",
            r"\dots",
            r"\\underline",
            r"\\tilde",
            r"\stackrel",
            r"\\frown",
            r"\widetilde",
            r"\widehat",
            r"\ddddot",
            r"\dddot",
            r"\\vec",
            r"\\breve",
            r"\chec",
            r"\overline",
            r"\overleftarrow",
            r"\overrightarrow",
            r"\mathring",
            r"\\not",
            r"\ddot",
            r"\dot",
            r"\\acute",
            r"\grave",
            r"\\bar",
            r"\hat",
            r"\colon",
            r"\\usepackage",
            r"\textstyle",
            r"\mathbin",
            r"\color",
            r"\newcommand",
            r"\ddots",
            r"\\vdots",
            r"\textrm",
            r"\textit",
            r"\textbf",
            r"\text",
            r"\\boldsymbol",
            r"\displaymath",
            r"\mathscr",
            r"\mathbb",
            r"\mathcal",
            r"\mathfrak",
            r"\mathtt",
            r"\mathsf",
            r"\mathbf",
            r"\mathit",
            r"\mathrm",
            r"\mathnormal",
            r"\\forall",
            r"\in",
            r"\quad",
            r"\exists",
            r"\leq",
            r"\epsilon",
            r"\\alpha",
            r"\Alpha",
            r"\\beta",
            r"\Beta",
            r"\gamma",
            r"\Gamma",
            r"\pi",
            r"\Pi",
            r"\phi,",
            r"\\varphi",
            r"\mu",
            r"\Phi",
            r"\cos",
            r"\theta",
            r"\cos",
            r"\theta",
            r"\sin",
            r"\theta",
            r"\\bmod",
            r"\equiv",
            r"\pmod",
            r"\\binom",
            r"\\frac",
            r"\rfrac",
            r"\times",
            r"\sfrac",
            r"\sqrt",
            r"\displaystyle",
            r"\idotsint",
            r"\iint",
            r"\\bigwedge",
            r"\\biguplus",
            r"\\bigodot",
            r"\coprod",
            r"\iiiint",
            r"\oint",
            r"\\bigvee",
            r"\\bigcup",
            r"\\bigoplus",
            r"\sum",
            r"\mathrm",
            r"\langle",
            r"\rangle",
            r"\lfloor",
            r"\rfloor",
            r"\lceil",
            r"\rceil",
            r"\\ulcorner",
            r"\\urcorner",
            r"\\backslash",
            r"\\big",
            r"\Big",
            r"\\bigg",
            r"\Bigg",
            r"\\bigcap",
            r"\\bigotimes",
            r"\prod",
            r"\iiint",
            r"\int",
            r"\\bigsqcup",
            r"\substack",
            r"\let",
            r"\oldsqrt",
            r"\def",
            r"\sqrt",
            r"\mathpalette",
            r"\DHLhksqrt",
            r"\setbox",
            r"\hbox",
            r"\oldsqrt",
            r"\dimen",
            r"\ht",
            r"\\advance",
            r"\\vrule",
            r"\lower",
            r"\\box",
            r"\\usepackage",
            r"\makeatletter",
            r"\let",
            r"\oldr",
            r"\LetLtxMacro",
            r"\renewcommand",
            r"\makeatother",
            "thm}",
            "prop}",
        ]

        self.latex_next_line = [r"\r", r"\n", r"\\n"]
        self.latex_commands_withoutbackslash = [
            repr(ext).split("\\")[-1][:-1]
            for ext in self.latex_commands + self.latex_math_commands
        ]
        self.latex_symbols = [
            r"\@",
            r"\,",
            r"\;",
            r"\:",
            r"\!",
            r"\-",
            r"\=",
            r"\>",
            r"\<",
            r"\+",
            r"\'",
            r"\`",
            r"\|",
            r"\(",
            r"\)",
            r"\[ ",
            r"\]",
            r"\\",
        ]
        self.latex_environments = [
            "{abstract}",
            "{array}",
            "{article}",
            "{center}",
            "{description}",
            "{displaymath}",
            "{document}",
            "{enumerate}",
            "{eqnarray}",
            "{equation}",
            "{example}",
            "{figure}",
            "{flushleft}",
            "{flushright}",
            "{itemize}",
            "{list}",
            "{math}",
            "{minipage}",
            "{picture}",
            "{quotation}",
            "{quote}",
            "{tabbing}",
            "{table}",
            "{tabular}",
            "{thebibliography}",
            "{theorem}",
            "{titlepage}",
            "{trivlist}",
            "{verbatim}",
            "{verse}",
            "{math}",
            "{pmatrix}",
            "{bmatrix}",
            "{matrix}",
            "{smallmatrix}",
            "{amsmath}",
            "{displaymath}",
            "{thm}",
            "{prop}",
            "{nakamoto}",
            "{visa}",
            "{wikicontracts}",
            "{bitcoinjmicropay}",
            "{amikopay}",
            "{impulse}",
            "{akselrod}",
            "{akselrod2}",
            "{todd}",
            "{not}",
            "{lamportpaxos}",
            "{lamportclocks}",
            "{exchange inputs:}",
            "{seqnum}",
            "{csv}",
            "{relcltv}",
            "{gregtimestop}",
            "{alicef}",
            "{bobf}",
            "{p2sh}",
            "{aliced}",
            "{bobd}",
            "{alicersmc2}",
            "{bobrsmc2}",
            "{bobrsmc2}",
            "{alicersmc2}",
            "{bip32}",
            "{smartcontracts}",
        ]
        self.latex_environments_withoutbracket = [
            ext[1:-1] for ext in self.latex_environments
        ]
        self.breaks = [
            "\r",
            "\r\n",
            "\x1c",
            "\x1d",
            "\x1e",
            "\x85",
            "\v",
            "\x0b",
            "\f",
            "\x0c",
            "\u2028",
        ]
        self.paragraph_break = ["\u2029"]

        self.one_letter = list(string.ascii_uppercase) + list(string.ascii_lowercase)
        self.roman_letter = list(self.roman_letters.values()) + [
            value.lower() for value in list(self.roman_letters.values())
        ]
        self.digit_letter = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        self.section_mark = ["§"]
        self.parenthesis = ["(", ")", "{", "}", "[", "]"]
        self.parenthesis_instart = ["(", "{", "["]
        self.parenthesis_inend = [")", "}", "]"]
        self.already_used_punctuations_instart = (
            self.section_mark + ["."] + ["*"] + self.parenthesis_instart
        )  # section_mark + dot + asterisk + parenthesis
        self.already_used_punctuations_inend = [":"] + self.parenthesis_inend  # colon
        self.already_used_punctuations = (
            self.already_used_punctuations_instart
            + self.already_used_punctuations_inend
        )

        self.bullet_list = ["\u2022", "•"]

        # Spacy punctuations:
        self.split_chars = lambda char: list(char.strip().split(" "))
        self._punct = r"… …… , : ; \! \? ¿ ؟ ¡ \( \) \[ \] \{ \} < > _ # \* & 。 ？ ！ ， 、 ； ： ～ · । ، ۔ ؛ ٪"
        self._quotes = (
            r'\' " ” “ ` ‘ ´ ’ ‚ , „ » « 「 」 『 』 （ ） 〔 〕 【 】 《 》 〈 〉'
        )
        self._hyphens = "- – — -- --- —— ~"
        self._currency = (
            r"\$ £ € ¥ ฿ US\$ C\$ A\$ ₽ ﷼ ₴ ₠ ₡ ₢ ₣ ₤ ₥ ₦ ₧ ₨ ₩ ₪ ₫ € ₭ ₮ ₯ ₰ "
            r"₱ ₲ ₳ ₴ ₵ ₶ ₷ ₸ ₹ ₺ ₻ ₼ ₽ ₾ ₿"
        )
        self._units = (
            "km km² km³ m m² m³ dm dm² dm³ cm cm² cm³ mm mm² mm³ ha µm nm yd in ft "
            "kg g mg µg t lb oz m/s km/h kmh mph hPa Pa mbar mb MB kb KB gb GB tb "
            "TB T G M K % км км² км³ м м² м³ дм дм² дм³ см см² см³ мм мм² мм³ нм "
            "кг г мг м/с км/ч кПа Па мбар Кб КБ кб Мб МБ мб Гб ГБ гб Тб ТБ тб"
            "كم كم² كم³ م م² م³ سم سم² سم³ مم مم² مم³ كم غرام جرام جم كغ ملغ كوب اكواب"
        )
        self.bar_upper = r"Ā B̄ C̄ D̄ Ē F̄ Ḡ H̄ Ī J̄ K̄ L̄ M̄ N̄ Ō P̄ Q̄ R̄ S̄ T̄ Ū V̄ W̄ X̄ Ȳ Z̄"

        self.LIST_spacy_PUNCT = self.split_chars(self._punct)
        self.LIST_spacy_QUOTES = self.split_chars(self._quotes)
        self.LIST_spacy_HYPHENS = self.split_chars(self._hyphens)
        self.LIST_spacy_CURRENCY = self.split_chars(self._currency)
        self.LIST_latex_bar_upper = self.split_chars(self.bar_upper)
        self.LIST_latex_bar_lower = [char.lower() for char in self.LIST_latex_bar_upper]
        self.LIST_latex_bar = self.LIST_latex_bar_upper + self.LIST_latex_bar_lower
        self.one_letter_bar_dict = {}
        for _, letter in enumerate(self.one_letter):
            self.one_letter_bar_dict[letter] = self.LIST_latex_bar[_]

        # All punctuations
        self.all_punctuations = list(
            set(
                self.punctuations
                + self.latex_symbol_letters
                + self.LIST_spacy_PUNCT
                + self.LIST_spacy_CURRENCY
                + self.LIST_spacy_QUOTES
                + self.LIST_spacy_HYPHENS
            )
        )

        # remove intersection between all punctuations and already_used_punctuations
        self.final_punctuations = list(
            set(self.all_punctuations) - set(self.already_used_punctuations)
        ) + list(set(self.already_used_punctuations) - set(self.all_punctuations))

        self.xml_special_charset = {
            "&quot;": '"',
            "&amp;": "&",
            "&lt;": "<",
            "&gt;": ">",
            "&OElig;": "Œ",
            "&oelig;": "œ",
            "&Scaron;": "Š",
            "&scaron;": "š",
            "&Yuml;": "Ÿ",
            "&circ;": "ˆ",
            "&tilde;": "˜",
            "&ndash;": "–",
            "&mdash;": "—",
            "&lsquo;": "‘",
            "&rsquo;": "’",
            "&sbquo;": "‚",
            "&ldquo;": "“",
            "&rdquo;": "”",
            "&bdquo;": "„",
            "&#8224;": "†",
            "&#8225;": "‡",
            "&#8240;": "‰",
            "&lsaquo;": "‹",
            "&rsaquo;": "›",
            "&euro;": "€",
        }

        self.rep = dict((re.escape(k), v) for k, v in self.xml_special_charset.items())


def drop_extra_columns(df, columns):

    for column in columns:
        try:
            df = df.drop(columns=[column], axis=1)
        except:
            continue
    return df


def remove_latex_commands_(text_line, especific_list):
    all_founded = []
    for (
        especific
    ) in (
        especific_list
    ):  # latex_commands+latex_math_commands+latex_symbols+paragraph_break+breaks:
        TEXTO = especific
        try:
            re_finds = re.findall(rf"\{TEXTO}", text_line)
        except:
            re_finds = re.findall(rf"{TEXTO}", text_line)
        for finded in re_finds:
            all_founded.append(finded)

    all_founded = sorted(
        list(set(all_founded)), key=len, reverse=True
    )  # sorted based on length

    not_all_char = []
    atleast_one_char = []
    for allfounded in all_founded:
        if any(char in Symbols().one_letter for char in allfounded):
            atleast_one_char.append(allfounded)
        else:
            not_all_char.append(allfounded)

    for founded in atleast_one_char + not_all_char:
        text_line = str(text_line).replace(founded, "")
    return text_line


def remove_especific_list_(text_line, especific_list):
    for especific in especific_list:
        text_line = str(text_line).replace(especific, "")
    return text_line


def remove_latex_environments_(text_line, especific_list):
    for especific in especific_list:
        text_line = str(text_line).replace(especific, "")
    return text_line


def is_compound_(word):
    """Checks if the word is in the list_of_specifics or is.digit() or is_compound_with_dot()
    input is a single word
    output is boolean"""

    compound_flag = False
    word = word.strip().replace(" ", ".")

    list_of_specifics = Symbols().section_mark + Symbols().parenthesis_instart

    if (
        is_in_a_specific_list(word.strip(), list_of_specifics)
        or is_roman_(word.strip(), Symbols().roman_letter)
        or is_single_letter_(word.strip(), Symbols().one_letter)
        or word.strip().isdigit()
        or is_compound_with_dot_(word) == True
    ):
        compound_flag = True
    return True if compound_flag == True else False


def is_compound_with_dot_(word):
    """Checks if the word is_compound_with_dot like: 1.1 , 1.A , A.a, ii.a, 1.i
    Checks any combination between is_digit_(), is_roman_(), is_single_letter_()
    input is a single word
    output is boolean"""

    if (
        "..." in word
    ):  # avoid "0 ....................................................." to be considered as compound
        return False
    if (
        counter_(word, Symbols().parenthesis_instart) == False
        or counter_(word, Symbols().parenthesis_inend) == False
    ):  # avoid "{(2)","U.S.C. 40103(b)(3)." to be considered as compound
        return False
    if (
        any(ext in word for ext in Symbols().parenthesis)
        and check_symmetry_of_parenthesis(word.strip()) == False
    ):  # avoid "[","(2" to be considered as compound
        return False

    list_of_specifics = Symbols().section_mark + [
        ""
    ]  # section_mark + parenthesis + [""]

    word = word.strip().replace(" ", ".")

    summ_digit = 0
    summ_punc = 0

    new_word = word

    for char in word:
        if char.isdigit():
            summ_digit = summ_digit + 1
        if is_punctuation_(char, Symbols().all_punctuations):
            summ_punc = summ_punc + 1
            if is_in_a_specific_list(
                char, Symbols().section_mark + Symbols().parenthesis
            ):
                new_word = new_word.replace(char, "." + char + ".")
                summ_punc = summ_punc + 2

    word = removeConsecutiveDuplicates(new_word)
    summ_dot = word.count(".")
    summ_punc = summ_punc - (len(new_word) - len(word))
    len(word) - summ_punc - summ_digit

    return_value = False

    if len(word.strip()) > 1:

        # if summ_digit==summ_char==summ_dot==0:
        #   return_value = False

        # if summ_digit == 0 and summ_dot!=0:
        #   if summ_char==summ_dot or summ_char-1==summ_dot  or summ_char+1==summ_dot:
        #     return_value = True

        # if  summ_char == 0 and summ_dot!=0:
        #   if summ_digit==summ_dot or summ_digit-1==summ_dot or summ_digit+1==summ_dot :
        #     return_value = True

        # if summ_digit!=0 and summ_char!=0 and summ_dot!=0:
        #   if (summ_char+summ_digit)==summ_dot or (summ_char+summ_digit)==summ_dot-1 or (summ_char+summ_digit)==summ_dot+1:
        #     return_value = True

        if len(word.strip().split(".")) > 1 and summ_dot != 0:  # if it's like ii.b
            if all(
                is_in_a_specific_list(ext.strip(), Symbols().parenthesis)
                or is_in_a_specific_list(ext.strip(), list_of_specifics)
                or is_roman_(ext.strip(), Symbols().roman_letter)
                or is_single_letter_(ext.strip(), Symbols().one_letter)
                or ext.strip().isdigit()
                for ext in word.strip().split(".")
            ):
                return_value = True

    return return_value


def data_preparation(text):
    header = text
    header = (
        header.lower().replace(" ", "").replace("\n", "").replace("", "").casefold()
    )
    for punct in Symbols().final_punctuations:
        header = header.replace(punct.casefold(), "")
    return header.casefold()


def is_roman_(word, roman_list):
    """Checks if a word is in the roman_list
    input is a single word
    output is boolean"""

    return True if word in roman_list else False


def is_single_letter_(word, single_letter_list):
    """Checks if a word is in the single_letter_list
    input is a single word
    output is boolean"""

    return True if word in single_letter_list else False


def is_in_a_specific_list(word, specific_list):
    """Checks if a word is in the specific_list
    input is a single word
    output is boolean"""

    return True if word in specific_list else False


def is_compound_(word):
    """Checks if the word is in the list_of_specifics or is.digit() or is_compound_with_dot()
    input is a single word
    output is boolean"""

    compound_flag = False
    word = word.strip().replace(" ", ".")

    list_of_specifics = Symbols().section_mark + Symbols().parenthesis_instart

    if (
        is_in_a_specific_list(word.strip(), list_of_specifics)
        or is_roman_(word.strip(), Symbols().roman_letter)
        or is_single_letter_(word.strip(), Symbols().one_letter)
        or word.strip().isdigit()
        or is_compound_with_dot_(word) == True
    ):
        compound_flag = True
    return True if compound_flag == True else False


# Recursive Program to remove consecutive dots
def removeConsecutiveDuplicates(s):
    if len(s) < 2:
        return s
    if s[0] != s[1]:
        return s[0] + removeConsecutiveDuplicates(s[1:])
    if s[0] == s[1] and s[0] != ".":
        return s[0] + removeConsecutiveDuplicates(s[1:])
    return removeConsecutiveDuplicates(s[1:])


def check_symmetry_of_parenthesis(my_str):

    if "[" in my_str and "]" in my_str:
        return True
    if "{" in my_str and "}" in my_str:
        return True
    if "(" in my_str and ")" in my_str:
        return True
    else:
        return False


def counter_(value, alist):
    """counts number of times any element in alist is found in the value"""
    count_list = [value.count(element) for element in alist]
    return True if sum(count_list) <= 1 else False


def is_itemized(word):
    roman_letter = Symbols().roman_letter

    last_char = word.strip()[-1]
    all_word_except_last_char = word.strip()[0:-1]

    if last_char == ")" and (
        all_word_except_last_char in roman_letter or all_word_except_last_char.isdigit()
    ):
        return True
    else:
        return False


def is_punctuation_(character, punct_list):
    """Checks if a character is in the punct_list
    input is a single letter
    output is boolean"""

    return True if character in punct_list else False


def is_compound_with_dot_(word):
    """Checks if the word is_compound_with_dot like: 1.1 , 1.A , A.a, ii.a, 1.i
    Checks any combination between is_digit_(), is_roman_(), is_single_letter_()
    input is a single word
    output is boolean"""

    if (
        "..." in word
    ):  # avoid "0 ....................................................." to be considered as compound
        return False
    if (
        counter_(word, Symbols().parenthesis_instart) == False
        or counter_(word, Symbols().parenthesis_inend) == False
    ):  # avoid "{(2)","U.S.C. 40103(b)(3)." to be considered as compound
        return False
    if (
        any(ext in word for ext in Symbols().parenthesis)
        and check_symmetry_of_parenthesis(word.strip()) == False
    ):  # avoid "[","(2" to be considered as compound
        return False

    list_of_specifics = Symbols().section_mark + [
        ""
    ]  # section_mark + parenthesis + [""]

    word = word.strip().replace(" ", ".")

    summ_digit = 0
    summ_punc = 0

    new_word = word

    for char in word:
        if char.isdigit():
            summ_digit = summ_digit + 1
        if is_punctuation_(char, Symbols().all_punctuations):
            summ_punc = summ_punc + 1
            if is_in_a_specific_list(
                char, Symbols().section_mark + Symbols().parenthesis
            ):
                new_word = new_word.replace(char, "." + char + ".")
                summ_punc = summ_punc + 2

    word = removeConsecutiveDuplicates(new_word)
    summ_dot = word.count(".")
    summ_punc = summ_punc - (len(new_word) - len(word))
    len(word) - summ_punc - summ_digit

    return_value = False

    if len(word.strip()) > 1:

        # if summ_digit==summ_char==summ_dot==0:
        #   return_value = False

        # if summ_digit == 0 and summ_dot!=0:
        #   if summ_char==summ_dot or summ_char-1==summ_dot  or summ_char+1==summ_dot:
        #     return_value = True

        # if  summ_char == 0 and summ_dot!=0:
        #   if summ_digit==summ_dot or summ_digit-1==summ_dot or summ_digit+1==summ_dot :
        #     return_value = True

        # if summ_digit!=0 and summ_char!=0 and summ_dot!=0:
        #   if (summ_char+summ_digit)==summ_dot or (summ_char+summ_digit)==summ_dot-1 or (summ_char+summ_digit)==summ_dot+1:
        #     return_value = True

        if len(word.strip().split(".")) > 1 and summ_dot != 0:  # if it's like ii.b
            if all(
                is_in_a_specific_list(ext.strip(), Symbols().parenthesis)
                or is_in_a_specific_list(ext.strip(), list_of_specifics)
                or is_roman_(ext.strip(), Symbols().roman_letter)
                or is_single_letter_(ext.strip(), Symbols().one_letter)
                or ext.strip().isdigit()
                for ext in word.strip().split(".")
            ):
                return_value = True

    return return_value


def mergeDict(dict1, dict2):
    """Merge dictionaries and keep values of common keys in list"""

    dict3 = {**dict1, **dict2}

    for key, value in dict3.items():

        if key in dict1 and key in dict2:

            dict3[key] = value + dict1[key]

    return dict3


# Python function to merge overlapping Intervals in
# O(n Log n) time and O(1) extra space
# Source: geekforgeeks


def mergeIntervals(arr):

    # Sorting based on the increasing order
    # of the start intervals
    arr.sort(key=lambda x: x[0])

    # Stores index of last element
    # in output array (modified arr[])
    index = 0

    # Traverse all input Intervals starting from
    # second interval
    for i in range(1, len(arr)):

        # If this is not first Interval and overlaps
        # with the previous one, Merge previous and
        # current Intervals
        if arr[index][1] >= arr[i][0]:
            arr[index][1] = max(arr[index][1], arr[i][1])
        else:
            index = index + 1
            arr[index] = arr[i]

    #    print("The Merged Intervals are :", end=" ")
    #    for i in range(index+1):
    #        print(arr[i], end=" ")

    return arr[: index + 1]


def get_table_location(page: fitz.Page, debugTable=False):
    """
    Get the location of tables in page
    by finding horizontal lines with same length

    Parameters
    ----------
    page: page object of pdf

    Returns
    -------
    table_rects: rectangles that contain tables
    """

    # make a list of horizontal lines
    # each line is represented by y and length
    hor_lines = []
    ver_lines = []
    hor_lines2 = []
    ver_lines2 = []
    min_x = 10000000
    max_x = -1
    paths = page.get_drawings()
    # print(paths)
    for p in paths:
        for item in p["items"]:
            if item[0] == "l":  # this is a line item
                p1 = item[1]  # start point
                p2 = item[2]  # stop point

                #                if debugTable:
                #                    print(p1, p2)
                if round(abs(p1.y - p2.y)) <= 0.25:  # line horizontal?
                    hor_lines.append(
                        (round(p1.y + 0.5), round(p2.x - p1.x + 0.5))
                    )  # potential table delimiter
                    minx = min(p1.x, p2.x)
                    if minx < min_x:
                        min_x = minx
                    maxx = max(p1.x, p2.x)
                    if maxx > max_x:
                        max_x = maxx

                    hor_lines2.append(
                        (
                            round(p1.x + 0.5),
                            round(p1.y + 0.5),
                            round(p2.x + 0.5),
                            round(p2.y + 0.5),
                        )
                    )
                if round(abs(p1.x - p2.x)) <= 0.25:
                    ver_lines.append(
                        (
                            round(p1.x + 0.5),
                            round(p1.y + 0.5),
                            round(p2.x + 0.5),
                            round(p2.y + 0.5),
                        )
                    )
                ver_lines2.append(
                    (
                        round(p1.x + 0.5),
                        round(p1.y + 0.5),
                        round(p2.x + 0.5),
                        round(p2.y + 0.5),
                    )
                )
            if item[0] == "re":  # this is a rectangle item
                r = item[1]

    hor_lines_dict = defaultdict(list)
    for li in hor_lines2:
        hor_lines_dict[li[1]].append([li[0], li[2]])

    # new_hor_lines = []
    # for k,v in hor_lines_dict.items():
    #    new_hor_lines.append((k,round(v)))
    # find whether table exists by number of lines with same length > 3
    table_rects = []
    # sort the list for ensuring the correct group by same keys
    hor_lines.sort(key=lambda x: x[1])

    # getting the top-left point and bottom-right point of table
    for k, g in itertools.groupby(hor_lines, key=lambda x: x[1]):
        g = list(g)
        if len(g) >= 3:  # number of lines of table will always >= 3
            g.sort(key=lambda x: x[0])  # sort by y value
            top_left = fitz.Point(0, g[0][0])
            bottom_right = fitz.Point(page.rect.width, g[-1][0])
            table_rects.append(fitz.Rect(top_left, bottom_right))

    if len(table_rects) >= 1:
        # create tempList
        ver_lines.sort(key=lambda x: x[1])
        tempVList = []
        for t in ver_lines:
            if round(t[3]) > round(t[1]):
                tempVList.append([round(t[1]), round(t[3])])
            else:
                tempVList.append([round(t[3]), round(t[1])])

        if len(tempVList) == 0:
            return table_rects

        tablesVerticalExtents = mergeIntervals(tempVList)

        horTableExtents = defaultdict()
        for k, v in hor_lines_dict.items():
            horTableExtents[k] = mergeIntervals(v)

        if tablesVerticalExtents != None:
            len(tablesVerticalExtents)
            new_table_rects = []
            # OLD-way
            # for r in tablesVerticalExtents:
            #    min_y = r[0]
            #    max_y = r[1]
            #    top_left = fitz.Point(min_x, min_y)
            #    bottom_right = fitz.Point(max_x, max_y)
            #    new_table_rects.append(fitz.Rect(top_left, bottom_right))
            # OLD-way
            for r in tablesVerticalExtents:
                try:
                    min_y = r[0]
                    max_y = r[1]
                    top_left = fitz.Point(horTableExtents[min_y][0][0], min_y)
                    bottom_right = fitz.Point(horTableExtents[max_y][0][1], max_y)
                    new_table_rects.append(fitz.Rect(top_left, bottom_right))
                except KeyError:
                    continue
            if new_table_rects != None:
                return new_table_rects
            else:
                return table_rects
    return table_rects


def detect_one_lines(df):
    """detects if parts of a line are separated while reading by pymupdf and should be merged"""

    features_need_to_be_based_on_first_part = [
        "first_char_isdigit",
        "starts_with_roman",
        "first_word_isletter",
        "first_word_iscompound",
        "first_char_is_special_letter",
        "starts_with_special_word",
        "starts_with_figure_table",
        "starts_with_asterisk",
        "starts_with_open_parenthesis",
        "starts_with_close_parenthesis",
        "is_first_char_dot",
        "first_word_is_stop",
        "first_word_is_noun_or_verb",
    ]

    indexes_should_be_removed = []

    for _, row in df.iterrows():

        if (
            df.at[_, "next_line_space"] < 0.1
            and _ < len(df) - 1
            and (
                df.iloc[_ + 1]["ratio_of_major_font_family"]
                == df.at[_, "ratio_of_major_font_family"]
            )
            and (
                df.iloc[_ + 1]["ratio_of_major_font_size"]
                == df.at[_, "ratio_of_major_font_size"]
            )
        ):

            indexes_should_be_removed.append(_)

            # features_need_to_be_based_on_first_part
            for feature in features_need_to_be_based_on_first_part:
                df.at[_ + 1, feature] = df.iloc[_][feature]

            # features_need_to_be_modified

            if (
                str(df.iloc[_ + 1]["text"])
                .strip()
                .startswith(str(df.iloc[_]["text"]).strip())
            ):
                df.at[_ + 1, "text"] = df.iloc[_ + 1]["text"]

            elif (
                str(df.iloc[_]["text"])
                .strip()
                .endswith(str(df.iloc[_ + 1]["text"]).strip())
            ):
                df.at[_ + 1, "text"] = df.iloc[_]["text"]
            else:
                df.at[_ + 1, "text"] = str(df.iloc[_]["text"]) + str(
                    df.iloc[_ + 1]["text"]
                )  # " / " + str(df.iloc[_+1]["text"])

            if (
                df.iloc[_ + 1]["figure_table_is_inside_text"] == 1
                or df.iloc[_]["figure_table_is_inside_text"] == 1
            ):
                df.at[_ + 1, "figure_table_is_inside_text"] = 1

            df.at[_ + 1, "text_len_difference"] = (
                df.iloc[_]["text_len_difference"]
                + df.iloc[_ + 1]["text_len_difference"]
            )
            df.at[_ + 1, "next_text_len_difference"] = df.iloc[_ + 1][
                "next_text_len_difference"
            ]
            df.at[_ + 1, "previous_text_len_difference"] = df.iloc[_][
                "previous_text_len_difference"
            ]

            df.at[_ + 1, "number_dot"] = (
                df.iloc[_]["number_dot"] + df.iloc[_ + 1]["number_dot"]
            )

            df.at[_ + 1, "right_margin"] = df.iloc[_ + 1]["right_margin"]
            df.at[_ + 1, "left_margin"] = df.iloc[_]["left_margin"]

            if (
                df.iloc[_ + 1]["left_margin"] > 0.05
                and abs(df.iloc[_ + 1]["left_margin"] - df.iloc[_ + 1]["right_margin"])
                < 0.1
            ):
                pass
            df.at[_ + 1, "is_centered"] = 1

            df.at[_ + 1, "last_char_is_colon"] = df.iloc[_ + 1]["last_char_is_colon"]

            df.at[_ + 1, "next_line_space"] = df.iloc[_ + 1]["next_line_space"]
            df.at[_ + 1, "previous_line_space"] = df.iloc[_]["previous_line_space"]

            df.at[_ + 1, "next_line_page_number"] = df.iloc[_ + 1][
                "next_line_page_number"
            ]
            df.at[_ + 1, "previous_line_page_number"] = df.iloc[_][
                "previous_line_page_number"
            ]

            df.at[_ + 1, "number_of_nouns"] = (
                df.iloc[_]["number_of_nouns"] + df.iloc[_ + 1]["number_of_nouns"]
            ) / 2
            df.at[_ + 1, "number_of_verbs"] = (
                df.iloc[_]["number_of_verbs"] + df.iloc[_ + 1]["number_of_verbs"]
            ) / 2
            df.at[_ + 1, "number_of_punctuations"] = (
                df.iloc[_]["number_of_punctuations"]
                + df.iloc[_ + 1]["number_of_punctuations"]
            ) / 2
            df.at[_ + 1, "number_of_is_stop"] = (
                df.iloc[_]["number_of_is_stop"] + df.iloc[_ + 1]["number_of_is_stop"]
            ) / 2
            df.at[_ + 1, "number_of_is_alpha"] = (
                df.iloc[_]["number_of_is_alpha"] + df.iloc[_ + 1]["number_of_is_alpha"]
            ) / 2
            df.at[_ + 1, "number_of_entities"] = (
                df.iloc[_]["number_of_entities"] + df.iloc[_ + 1]["number_of_entities"]
            ) / 2
            df.at[_ + 1, "number_of_upper"] = (
                df.iloc[_]["number_of_upper"] + df.iloc[_ + 1]["number_of_upper"]
            ) / 2

            df.at[_ + 1, "paragraph_break"] = df.iloc[_ + 1]["paragraph_break"]
            df.at[_ + 1, "break"] = df.iloc[_ + 1]["break"]

            if df.iloc[_ + 1]["is_bold"] == 1 or df.iloc[_]["is_bold"] == 1:
                df.at[_ + 1, "is_bold"] = 1
            if df.iloc[_ + 1]["is_italic"] == 1 or df.iloc[_]["is_italic"] == 1:
                df.at[_ + 1, "is_italic"] = 1

            if (
                df.iloc[_ + 1]["multiple_font_size"] == 1
                or df.iloc[_]["multiple_font_size"] == 1
            ):
                df.at[_ + 1, "multiple_font_size"] = 1
            if (
                df.iloc[_ + 1]["multiple_font_family"] == 1
                or df.iloc[_]["multiple_font_family"] == 1
            ):
                df.at[_ + 1, "multiple_font_family"] = 1

            all_is_upper = 0
            if df.iloc[_ + 1]["all_is_upper"] == 1 and df.iloc[_]["all_is_upper"] == 1:
                all_is_upper = 1
            df.at[_ + 1, "all_is_upper"] = all_is_upper

            df.at[_ + 1, "distance_from_top"] = df.iloc[_]["distance_from_top"]
            df.at[_ + 1, "distance_from_bottom"] = df.iloc[_ + 1][
                "distance_from_bottom"
            ]

            has_bullet_point = 0
            if (
                df.iloc[_ + 1]["has_bullet_point"] == 1
                or df.iloc[_]["has_bullet_point"] == 1
            ):
                has_bullet_point = 1
            df.at[_ + 1, "has_bullet_point"] = has_bullet_point

            df.at[_ + 1, "is_last_word_roman_or_digit"] = df.iloc[_ + 1][
                "is_last_word_roman_or_digit"
            ]

            df.at[_ + 1, "ratio_of_major_font_family"] = df.iloc[_][
                "ratio_of_major_font_family"
            ]
            df.at[_ + 1, "ratio_of_major_font_size"] = df.iloc[_][
                "ratio_of_major_font_size"
            ]

            df.at[_ + 1, "text_total_number_of_extra_spaces"] = (
                df.iloc[_]["text_total_number_of_extra_spaces"]
                + df.iloc[_ + 1]["text_total_number_of_extra_spaces"]
            )

            df.at[_ + 1, "font_size"] = df.iloc[_]["font_size"]

            # if df.iloc[_+1]["ist_oc"]==1 or df.iloc[_]["is_toc"]==1:
            #     df.at[_+1,"is_toc"] = 1

            # if df.iloc[_+1]["is_header"]==1 or df.iloc[_]["is_header"]==1:
            #     df.at[_+1,"is_header"] = 1

            # if df.iloc[_+1]["is_firstline"]==1 or df.iloc[_]["is_firstline"]==1:
            #     df.at[_+1,"is_firstline"] = 1

            # if df.iloc[_+1]["is_table_column"]==1 or df.iloc[_]["is_table_column"]==1:
            #     df.at[_+1,"is_table_column"] = 1

            # if df.iloc[_+1]["is_table_title"]==1 or df.iloc[_]["is_table_title"]==1:
            #     df.at[_+1,"is_table_title"] = 1

            # if df.iloc[_+1]["is_table_content"]==1 or df.iloc[_]["is_table_content"]==1:
            #     df.at[_+1,"is_table_content"] = 1

            # try:
            #   df.at[_+1,"parent_tags"] = df.iloc[_+1]["parent_tags"]+"   &&   "+df.iloc[_]["parent_tags"]
            # except:
            #   df.at[_+1,"parent_tags"] = "no-tag"

    df = df.drop(df.index[indexes_should_be_removed]).reset_index()
    return df


def bin_constructor(column_list):
    """bin values of a column and convert it from continuous values to categorical"""

    data_col = list(sorted(column_list))

    mini = min(data_col)
    maxi = max(data_col)
    # print(column_list)
    data_col = data_col[
        int(0.01 * len(data_col)) : int(len(data_col) - 0.01 * len(data_col))
    ]

    # m = np.mean(data_col)
    m = statistics.mode(data_col)
    # s = np.std(data_col)
    s = np.nanstd(data_col)
    # print(m,s)

    first_bin_upper_bound = m - (2 * s)
    last_bin_lower_bound = m + (2 * s)

    if mini > first_bin_upper_bound:
        mini = first_bin_upper_bound

    if maxi < last_bin_lower_bound:
        maxi = last_bin_lower_bound

    bins = [
        I.closedopen(mini, first_bin_upper_bound),
        I.closedopen(m - (2 * s), m - (1 * s)),
        I.closedopen(m - (1 * s), m - (0.5 * s)),
        I.closedopen(m - (0.5 * s), m - (0.125 * s)),
        I.closedopen(m - (0.125 * s), m + (0.125 * s)),
        I.closedopen(m + (0.125 * s), m + (0.5 * s)),
        I.closedopen(m + (0.5 * s), m + (1 * s)),
        I.closedopen(m + (1 * s), m + (2 * s)),
        I.closed(last_bin_lower_bound, maxi),
    ]

    bins_keys = [integer for integer in range(1, len(bins) + 1)]

    new_column_list = []

    for _, value in enumerate(column_list):
        index = [_ for _, interval in enumerate(bins) if value in interval]
        if len(index) != 0:
            new_column_list.append(bins_keys[index[0]])
        else:
            new_column_list.append(np.nan)

    return new_column_list  # ,bins,bins_keys
