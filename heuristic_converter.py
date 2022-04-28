import random
import re
from collections import OrderedDict
from typing import Callable, Optional
from xml import dom


def safeint(x) -> int:
    try:
        return int(x)
    except:
        return 0


DOMAIN_PHRASE_IN_SENTENCE = {
    "관광": "관광",
    "숙소": "숙소",
    "식당": "식당",
    "지하철": "지하철",
    "택시": "택시",
}

DOMAIN_SLOT_TEMPLATES = {
    "관광": OrderedDict(
        [
            ("관광-경치 좋은", lambda x, either: f"경치 {either('좋지 않은' if x =='no' else '좋은')}"),
            ("관광-교육적", lambda x, either: f"교육 {either('관련 없는' if x == 'no' else '관련된')}"),
            ("관광-도보 가능", lambda x, either: f"도보로 갈 수 {either('없' if x == 'no' else '있')}는"),
            ("관광-문화 예술", lambda x, either: f"문화 예술 {either('관련 없는' if x == 'no' else '관련된')}"),
            ("관광-역사적", lambda x, either: f"역사 {either('관련 없는' if x == 'no' else '관련된')}"),
            ("관광-이름", lambda x, either: f"이름은 {either(x)}"),
            ("관광-종류", lambda x, either: f"종류는 {either(x)}"),
            ("관광-주차 가능", lambda x, either: f"차량 주차 {either('불' if x == 'no' else '')}가능한"),
            ("관광-지역", lambda x, either: f"위치는 {either(x)}"),
        ]
    ),
    "숙소": OrderedDict(
        [
            ("숙소-가격대", lambda x, either: f"가격 {either('비싼' if x == '비싼' else f'{x}한')}"),
            ("숙소-도보 가능", lambda x, either: f"도보로 갈 수 {either('없' if x == 'no' else '있')}는"),
            ("숙소-수영장 가능", lambda x, either: f"수영장 {either('없' if x == 'no' else '있')}는"),
            ("숙소-스파 유무", lambda x, either: f"스파 {either('없' if x == 'no' else '있')}는"),
            ("숙소-예약 기간", lambda x, either: f"예약 기간 {either(x)}"),
            ("숙소-예약 명수", lambda x, either: f"예약하는 사람의 수 {either(x)}"),
            ("숙소-예약 요일", lambda x, either: f"예약 날짜 {either(x)}"),
            ("숙소-이름", lambda x, either: f"이름은 {either(x)}"),
            ("숙소-인터넷 가능", lambda x, either: f"인터넷 사용 {either('불' if x == 'no' else '')}가능한"),
            ("숙소-조식 가능", lambda x, either: f"조식 {either('불' if x == 'no' else '')}가능한"),
            ("숙소-종류", lambda x, either: f"종류는 {either(x)}"),
            ("숙소-주차 가능", lambda x, either: f"차량 주차 {either('불' if x == 'no' else '')}가능한"),
            ("숙소-지역", lambda x, either: f"위치는 {either(x)}"),
            ("숙소-헬스장 유무", lambda x, either: f"헬스장 {either('없' if x == 'no' else '있')}는"),
            ("숙소-흡연 가능", lambda x, either: f"흡연 {either('불' if x == 'no' else '')}가능한"),
        ]
    ),
    "식당": OrderedDict(
        [
            ("식당-가격대", lambda x, either: f"가격 {either('' if x == '비싼' else '{x}한')}"),
            ("식당-도보 가능", lambda x, either: f"도보로 갈 수 {either('없' if x == 'no' else '있')}는"),
            ("식당-야외석 유무", lambda x, either: f"야외 자리 {either('없' if x == 'no' else '있')}는"),
            ("식당-예약 명수", lambda x, either: f"예약하는 사람의 수 {either(x)}"),
            ("식당-예약 시간", lambda x, either: f"예약 시간 {either(x)}"),
            ("식당-예약 요일", lambda x, either: f"예약 날짜 {either(x)}"),
            ("식당-이름", lambda x, either: f"이름은 {either(x)}"),
            ("식당-인터넷 가능", lambda x, either: f"인터넷 사용 {either('불' if x == 'no' else '')}가능한"),
            ("식당-종류", lambda x, either: f"종류는 {either(x)}"),
            ("식당-주류 판매", lambda x, either: f"주류 판매{either('하지 않' if x == 'no' else '하')}는"),
            ("식당-주차 가능", lambda x, either: f"차량 주차 {either('불' if x == 'no' else '')}가능한"),
            ("식당-지역", lambda x, either: f"위치는 {either(x)}"),
            ("식당-흡연 가능", lambda x, either: f"흡연 {either('불' if x == 'no' else '')}가능한"),
        ]
    ),
    "지하철": OrderedDict(
        [
            ("지하철-도착지", lambda x, either: f"도착지는 {either(x)}"),
            ("지하철-출발 시간", lambda x, either: f"출발 시간은 {either(x)}"),
            ("지하철-출발지", lambda x, either: f"출발지는 {either(x)}"),
        ]
    ),
    "택시": OrderedDict(
        [
            ("택시-도착 시간", lambda x, either: f"도착 시간은 {either(x)}"),
            ("택시-도착지", lambda x, either: f"도착지는 {either(x)}"),
            ("택시-종류", lambda x, either: f"종류는 {either(x)}"),
            ("택시-출발 시간", lambda x, either: f"출발 시간은 {either(x)}",),
            ("택시-출발지", lambda x, either: f"출발지는 {either(x)}"),
        ]
    ),
}

DOMAIN_DONTCARE_PHRASES_DICT = {
    "관광": OrderedDict(
        [
            ("관광-경치 좋은", "경치가 좋은지 나쁜지 상관없는"),
            ("관광-교육적", "교육과 관련이 있는지 없는지 상관없는"),
            ("관광-도보 가능", "도보로 갈 수 있는지 없는지 상관없는"),
            ("관광-문화 예술", "문화 예술과 관련이 있는지 없는지 상관없는"),
            ("관광-역사적", "역사와 관련이 있는지 없는지 상관없는"),
            ("관광-이름", "이름이 상관없는"),
            ("관광-종류", "종류가 상관없는"),
            ("관광-주차 가능", "차량 주차가 가능한지 불가능한지 상관없는"),
            ("관광-지역", "위치가 상관없는"),
        ]
    ),
    "숙소": OrderedDict(
        [
            ("숙소-가격대", "가격이 상관없는"),
            ("숙소-도보 가능", "도보로 갈 수 있는지 없는지 상관없는"),
            ("숙소-수영장 유무", "수영장이 있는지 없는지 상관없는"),
            ("숙소-스파 유무", "스파가 있는지 없는지 상관없는"),
            ("숙소-예약 기간", "예약하는 기간은 상관없는"),
            ("숙소-예약 명수", "예약하는 사람의 수는 상관없는"),
            ("숙소-예약 요일", "예약하는 날짜는 상관없는"),
            ("숙소-이름", "이름이 상관없는"),
            ("숙소-인터넷 가능", "인터넷 사용이 가능한지 불가능한지 상관없는"),
            ("숙소-조식 가능", "조식이 가능한지 불가능한지 상관없는"),
            ("숙소-종류", "종류가 상관없는"),
            ("숙소-주차 가능", "차량 주차가 가능한지 불가능한지 상관없는"),
            ("숙소-지역", "위치가 상관없는"),
            ("숙소-헬스장 유무", "헬스장이 있는지 없는지 상관없는"),
            ("숙소-흡연 가능", "흡연이 가능한지 불가능한지 상관없는"),
        ]
    ),
    "식당": OrderedDict(
        [
            ("식당-가격대", "가격이 상관없는"),
            ("식당-도보 가능", "도보로 갈 수 있는지 없는지 상관없는"),
            ("식당-야외석 유무", "야외에 자리가 있는지 없는지 상관없는"),
            ("식당-예약 명수", "예약 명수는 상관없는"),
            ("식당-예약 시간", "예약 시간은 상관없는"),
            ("식당-예약 요일", "예약 날짜는 상관없는"),
            ("식당-이름", "이름이 상관없는"),
            ("식당-인터넷 가능", "인터넷 사용이 가능한지 불가능한지 상관없는"),
            ("식당-종류", "종류가 상관없는"),
            ("식당-주류 판매", "주류를 판매하는지 하지않는지 상관없는"),
            ("식당-주차 가능", "차량 주차가 가능한지 불가능한지 상관없는"),
            ("식당-지역", "위치가 상관없는"),
            ("식당-흡연 가능", "흡연이 가능한지 불가능한지 상관없는"),
        ]
    ),
    "지하철": OrderedDict([("지하철-도착지", "도착지는 상관없는"), ("지하철-출발 시간", "출발 시간은 상관없는"), ("지하철-출발지", "출발지는 상관없는")]),
    "택시": OrderedDict(
        [
            ("택시-도착 시간", "도착 시간은 상관없는"),
            ("택시-도착지", "도착지는 상관없는"),
            ("택시-종류", "종류가 상관없는"),
            ("택시-출발 시간", "출발 시간은 상관없는"),
            ("택시-출발지", "출발지는 상관없는"),
        ]
    ),
}

COMMON_PHRASES = ["을 찾는다.", "를 찾는다."]


def check_consonant(domain: str):
    """Check the final consonant of the domain for natural heuristic converter.

    Args:
        domain (str): domains of the dataset in Korean.
    """
    k = DOMAIN_PHRASE_IN_SENTENCE[domain][-1]
    if "가" <= k <= "힣":
        return (ord(k) - ord("가")) % 28 > 0
    else:
        return


def get_summary_sentence(dialog_state: dict, domain: str, either: callable) -> str:
    # For example, if ds = {"관광-종류": "박물관", "관광-지역": "서울 북쪽", "관광-교육적": "yes"},
    # slot_phrases = {'관광-교육적': '교육적인', '관광-종류': '종류가 박물관인', '관광-지역': '서울 북쪽에 위치한'}
    slot_phrases = {
        _slot_name: f(dialog_state[_slot_name], either)
        for _slot_name, f in DOMAIN_SLOT_TEMPLATES[domain].items()
        if _slot_name in dialog_state and dialog_state[_slot_name] != "dontcare"
    }

    # example: '그는 교육적인, 종류가 박물관인, 서울 북쪽에 위치한 관광을 찾고 있다'
    is_consonant = check_consonant(domain)
    common_phrases_idx = 0 if is_consonant else 1
    first_sentence = "".join(["user는 "] + [_phrase + ", " for i, (_, _phrase) in enumerate(slot_phrases.items())])

    dontcare_sentence = get_dontcare_sentence(dialog_state=dialog_state, domain=domain, either=either)
    if dontcare_sentence != "":
        # use text slicing for `dontcare_sentence` to delete ','
        summary = (
            first_sentence
            + dontcare_sentence
            + " "
            + DOMAIN_PHRASE_IN_SENTENCE[domain]
            + COMMON_PHRASES[common_phrases_idx]
        )
    else:
        # use text slicing for `first_sentence` to delete ', '
        summary = first_sentence[:-2] + " " + DOMAIN_PHRASE_IN_SENTENCE[domain] + COMMON_PHRASES[common_phrases_idx]

    return summary


def get_dontcare_sentence(dialog_state: dict, domain: str, either: callable):
    # dontcare_phrases (list(str)): phrases corresponding to slot name with value of dontcare
    # for example, `dontcare_phrases` is ['종류는'] if ds = {'식당-종류': 'dontcare'}
    dontcare_phrases = [
        either(_phrase)
        for _slot_name, _phrase in DOMAIN_DONTCARE_PHRASES_DICT[domain].items()
        if _slot_name in dialog_state and dialog_state[_slot_name] == "dontcare"
    ]

    # in case of `doncare_phrases` exists
    if dontcare_phrases:
        dontcare_sentence = dontcare_phrases[0]
    else:
        return ""

    return dontcare_sentence


def get_dontcare_values(summary: str, domain: str):
    """Find the slot name with the value of dontcare in the summary."""
    return {
        _slot_name: "dontcare"
        for _slot_name, _phrase in DOMAIN_DONTCARE_PHRASES_DICT[domain].items()
        if re.search(_phrase, summary)
    }


def convert_tour_state_to_summary(dialog_state: dict, either: callable) -> str:
    """Convert dialogue state to summary for tour domain."""
    summary = get_summary_sentence(dialog_state=dialog_state, domain="관광", either=either)

    return summary


def convert_hotel_state_to_summary(dialog_state: dict, either: callable) -> str:
    summary = get_summary_sentence(dialog_state=dialog_state, domain="숙소", either=either)

    return summary


def convert_restaurant_state_to_summary(dialog_state: dict, either: callable) -> str:
    summary = get_summary_sentence(dialog_state=dialog_state, domain="식당", either=either)

    return summary


def convert_subway_state_to_summary(dialog_state: dict, either: callable) -> str:
    summary = get_summary_sentence(dialog_state=dialog_state, domain="지하철", either=either)

    return summary


def convert_taxi_state_to_summary(dialog_state: dict, either: callable) -> str:
    summary = get_summary_sentence(dialog_state=dialog_state, domain="택시", either=either)

    return summary


def convert_tour_summary_to_state(summ: str) -> dict:
    sentences = str.split(summ, sep=". ")
    summary = [sentence for sentence in sentences if DOMAIN_PHRASE_IN_SENTENCE["관광"] in sentence]
    if not summary:
        return {}
    summary = summary[0]

    slot_to_prefix = {
        "관광-경치 좋은": "경치",
        "관광-교육적": "교육",
        "관광-도보 가능": "도보로 갈 수",
        "관광-문화 예술": "문화 예술",
        "관광-역사적": "역사",
        "관광-이름": "이름은",
        "관광-종류": "종류는",
        "관광-주차 가능": "차량 주차",
        "관광-지역": "위치는",
    }

    # set aside for slots with value of dontcare
    dontcare_sentence = summary

    temp_state = {}
    for slot, prefix in slot_to_prefix.items():
        # check prefixes are in summary
        match = re.search(prefix, summary)
        if match:
            start_idx = match.span()[-1]
            value = re.split(pattern="는|된|,| 관광", string=summary[start_idx:])[0]
            # get temporary state made up of phrase-type values
            temp_state[slot] = value.replace(",", "").replace(".", "").strip()

    tour_state = {}
    for _slot, _phrase in temp_state.items():
        if _slot not in ["관광-이름", "관광-종류", "관광-지역"]:
            if _phrase == "좋지 않은":
                tour_state[_slot] = "no"
            if _phrase == "좋은":
                tour_state[_slot] = "yes"
            if _phrase == "관련 없":
                tour_state[_slot] = "no"
            if _phrase == "관련":
                tour_state[_slot] = "yes"
            if _phrase == "없":
                tour_state[_slot] = "no"
            if _phrase == "있":
                tour_state[_slot] = "yes"
            if _phrase == "불가능한":
                tour_state[_slot] = "no"
            if _phrase == "가능한":
                tour_state[_slot] = "yes"
            if "불가능한" in _phrase:
                tour_state[_slot] = "no"
                continue
            if "가능한" in _phrase:
                tour_state[_slot] = "yes"
        else:
            if _phrase != "상관없":
                tour_state[_slot] = _phrase

    # get dontcare value and update
    tour_state.update(get_dontcare_values(summary=dontcare_sentence, domain="관광"))

    return tour_state


def convert_restaurant_summary_to_state(summ: str) -> dict:
    sentences = str.split(summ, sep=". ")
    summary = [sentence for sentence in sentences if DOMAIN_PHRASE_IN_SENTENCE["식당"] in sentence]
    if not summary:
        return {}
    summary = summary[0]

    slot_to_prefix = {
        "식당-가격대": "가격",
        "식당-도보 가능": "도보로 갈 수",
        "식당-야외석 유무": "야외 자리",
        "식당-예약 명수": "예약하는 사람의 수",
        "식당-예약 시간": "예약 시간",
        "식당-예약 요일": "예약 날짜",
        "식당-이름": "이름은",
        "식당-인터넷 가능": "인터넷 사용",
        "식당-종류": "종류는",
        "식당-주류 판매": "주류 판매",
        "식당-주차 가능": "차량 주차",
        "식당-지역": "위치는",
        "식당-흡연 가능": "흡연",
    }

    # set aside for slots with value of dontcare
    dontcare_sentence = summary

    temp_state = {}
    for slot, prefix in slot_to_prefix.items():
        match = re.search(prefix, summary)
        if match:
            start_idx = match.span()[-1]
            value = re.split("는|싼|,| 식당", summary[start_idx:])[0]
            temp_state[slot] = value.replace(",", "").replace(".", "").strip()

    restaurant_state = {}
    for _slot, _phrase in temp_state.items():
        if _slot not in ["식당-예약 명수", "식당-예약 시간", "식당-예약 요일", "식당-이름", "식당-종류", "식당-지역"]:
            if _phrase == "없":
                restaurant_state[_slot] = "no"
            if _phrase == "있":
                restaurant_state[_slot] = "yes"
            if _phrase == "불가능한":
                restaurant_state[_slot] = "no"
            if _phrase == "가능한":
                restaurant_state[_slot] = "yes"
            if _phrase == "비":
                restaurant_state[_slot] = "비싼"
            if _phrase == "저렴한":
                restaurant_state[_slot] = "저렴"
            if _phrase == "적당한":
                restaurant_state[_slot] = "적당"
            if _phrase == "하지 않":
                restaurant_state[_slot] = "no"
            if _phrase == "하":
                restaurant_state[_slot] = "yes"
            if "불가능한" in _phrase:
                restaurant_state[_slot] = "no"
                continue
            if "가능한" in _phrase:
                restaurant_state[_slot] = "yes"
        else:
            if _phrase != "상관없":
                restaurant_state[_slot] = _phrase

    # get dontcare value and update
    restaurant_state.update(get_dontcare_values(summary=dontcare_sentence, domain="식당"))

    return restaurant_state


def convert_hotel_summary_to_state(summ: str) -> dict:
    sentences = str.split(summ, sep=". ")
    summary = [sentence for sentence in sentences if DOMAIN_PHRASE_IN_SENTENCE["숙소"] in sentence]
    if not summary:
        return {}
    summary = summary[0]

    slot_to_prefix = {
        "숙소-가격대": "가격",
        "숙소-도보 가능": "도보로 갈 수",
        "숙소-수영장 가능": "수영장",
        "숙소-스파 유무": "스파",
        "숙소-예약 기간": "예약 기간",
        "숙소-예약 명수": "예약하는 사람의 수",
        "숙소-예약 요일": "예약 날짜",
        "숙소-이름": "이름은",
        "숙소-인터넷 가능": "인터넷 사용",
        "숙소-조식 가능": "조식",
        "숙소-종류": "종류는",
        "숙소-주차 가능": "차량 주차",
        "숙소-지역": "위치는",
        "숙소-헬스장 유무": "헬스장",
        "숙소-흡연 가능": "흡연",
    }

    dontcare_sentence = summary

    temp_state = {}
    for slot, prefix in slot_to_prefix.items():
        match = re.search(prefix, summary)
        if match:
            start_idx = match.span()[-1]
            value = re.split("는|싼|,| 숙소", summary[start_idx:])[0]
            temp_state[slot] = value.replace(",", "").replace(".", "").strip()

    hotel_state = {}
    for _slot, _phrase in temp_state.items():
        if _slot not in ["숙소-예약 명수", "숙소-예약 요일", "숙소-예약 기간", "숙소-이름", "숙소-종류", "숙소-지역"]:
            if _phrase == "없":
                hotel_state[_slot] = "no"
            if _phrase == "있":
                hotel_state[_slot] = "yes"
            if _phrase == "불가능한":
                hotel_state[_slot] = "no"
            if _phrase == "가능한":
                hotel_state[_slot] = "yes"
            if _phrase == "비":
                hotel_state[_slot] = "비싼"
            if _phrase == "저렴한":
                hotel_state[_slot] = "저렴"
            if _phrase == "적당한":
                hotel_state[_slot] = "적당"
            if "불가능한" in _phrase:
                hotel_state[_slot] = "no"
                continue
            if "가능한" in _phrase:
                hotel_state[_slot] = "yes"
        else:
            if _phrase != "상관없":
                hotel_state[_slot] = _phrase

    hotel_state.update(get_dontcare_values(summary=dontcare_sentence, domain="숙소"))

    return hotel_state


def convert_subway_summary_to_state(summ: str) -> dict:
    sentences = str.split(summ, sep=". ")
    summary = [sentence for sentence in sentences if DOMAIN_PHRASE_IN_SENTENCE["지하철"] in sentence]
    if not summary:
        return {}
    summary = summary[0]

    slot_to_prefix = {"지하철-도착지": "도착지는", "지하철-출발 시간": "출발 시간은", "지하철-출발지": "출발지는"}

    dontcare_sentence = summary

    temp_state = {}
    for slot, prefix in slot_to_prefix.items():
        match = re.search(prefix, summary)
        if match:
            start_idx = match.span()[-1]
            value = re.split(",| 지하철", summary[start_idx:])[0]
            temp_state[slot] = value.replace(",", "").replace(".", "").strip()

    subway_state = {}
    for _slot, _phrase in temp_state.items():
        if _phrase != "상관없는":
            subway_state[_slot] = _phrase

    subway_state.update(get_dontcare_values(summary=dontcare_sentence, domain="지하철"))

    return subway_state


def convert_taxi_summary_to_state(summ: str) -> dict:
    sentences = str.split(summ, sep=". ")
    summary = [sentence for sentence in sentences if DOMAIN_PHRASE_IN_SENTENCE["택시"] in sentence]
    if not summary:
        return {}
    summary = summary[0]

    slot_to_prefix = {"택시-도착 시간": "도착 시간은", "택시-도착지": "도착지는", "택시-종류": "종류는", "택시-출발 시간": "출발 시간은", "택시-출발지": "출발지는"}

    dontcare_sentence = summary

    temp_state = {}
    for slot, prefix in slot_to_prefix.items():
        match = re.search(prefix, summary)
        if match:
            start_idx = match.span()[-1]
            value = re.split(",| 택시", summary[start_idx:])[0]
            temp_state[slot] = value.replace(",", "").replace(".", "").strip()

    taxi_state = {}
    for _slot, _phrase in temp_state.items():
        if _phrase != "상관없는":
            taxi_state[_slot] = _phrase

    taxi_state.update(get_dontcare_values(summary=dontcare_sentence, domain="택시"))

    return taxi_state


class KluewosConverter:
    def __init__(self, wo_para: bool = False, do_concat: bool = True):
        self.domain_state_to_summ: dict[str, Callable] = {
            "관광": convert_tour_state_to_summary,
            "숙소": convert_hotel_state_to_summary,
            "식당": convert_restaurant_state_to_summary,
            "지하철": convert_subway_state_to_summary,
            "택시": convert_taxi_state_to_summary,
        }
        self.domain_summ_to_state = {
            "관광": convert_tour_summary_to_state,
            "숙소": convert_hotel_summary_to_state,
            "식당": convert_restaurant_summary_to_state,
            "지하철": convert_subway_summary_to_state,
            "택시": convert_taxi_summary_to_state,
        }
        self.do_concat = do_concat
        self.wo_para = wo_para

    def convert_summary_to_state(self, summ: str) -> dict:
        state_dict = {}
        for _sum_to_state in self.domain_summ_to_state.values():
            state_dict.update(_sum_to_state(summ=summ))

        return state_dict

    def convert_state_to_summary(
        self, dialog_state: dict, is_for_template: Optional[bool] = False, blank: Optional[str] = None
    ) -> str:
        appearing_domains = list(set(k.split("-")[0] for k in dialog_state) & set(self.domain_state_to_summ.keys()))
        random.shuffle(appearing_domains)
        either = lambda x: blank if is_for_template else x

        sentences = [
            self.domain_state_to_summ[domain](dialog_state=dialog_state, either=either) for domain in appearing_domains
        ]
        # when list is length 1, join does not add '그리고', and just returns an str
        summary = " 그리고 ".join(sentences)

        return summary


class DomainFreeConverter:
    def __init__(self):
        self.sentence_prefix = "The user wants "
        self.slot_prefix = " as "
        self.domain_prefix = " of "
        self.phrase_divider = ", "
        self.sentence_postfix = "."

    def state_to_sum(
        self, ds: dict, is_for_template: Optional[bool] = False, blank: Optional[str] = None, is_one_sentence=True
    ) -> str:
        """
        If we wants to generate various templates and lm ranking, we could fit better preposition for each slot
        Input:
            example: {'domain-key1': 'value1', 'key2': 'value2'}
        Returns:
            example: "The user wants key1 as value1, key2 as value2"
            real_ex: "The user wants london as departure, cambridge as destination, 12:30 as arriveby, 3 as book people,
                    tuesday as day."
        """
        res = self.sentence_prefix
        for i, (domain_slot, value) in enumerate(ds.items()):
            if i > 0:
                res += self.phrase_divider
            domain = domain_slot.split("-")[0]
            slot = domain_slot.split("-")[-1]
            phrase = value + self.slot_prefix + slot + self.domain_prefix + domain
            res += phrase

        res += self.sentence_postfix
        return res

    def sum_to_state(self, summary: str) -> dict:
        res = {}
        summary = summary.replace(self.sentence_prefix, "")
        summary = summary.replace(self.sentence_postfix, "")
        summary = summary.split(self.phrase_divider)
        for phrase in summary:
            if self.domain_prefix not in phrase or self.slot_prefix not in phrase:
                continue
            value = phrase.split(self.slot_prefix)[0]
            slot_of_domain = phrase.split(self.slot_prefix)[-1]
            slot = slot_of_domain.split(self.domain_prefix)[0]
            domain = slot_of_domain.split(self.domain_prefix)[-1]
            res[f"{domain}-{slot}"] = value
        return res


def get_converter(converter_name: str):
    # without paraphrasing
    if converter_name == "wo_para":
        return KluewosConverter(wo_para=True, do_concat=True)
    # without one sentence concatenating
    if converter_name == "wo_concat":
        return KluewosConverter(wo_para=False, do_concat=False)
    if converter_name == "open_domain":
        return DomainFreeConverter()
    if converter_name == "vanilla":
        return KluewosConverter(wo_para=True, do_concat=False)

    return KluewosConverter(wo_para=False, do_concat=True)
