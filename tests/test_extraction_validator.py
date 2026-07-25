from datagen.extraction_validator import validate_extraction

CLEAN_ARABIC = (
    "هذا نص تجريبي يستخدم لاختبار جودة الاستخراج. "
    "يحتوي هذا النص على جمل عربية طبيعية بمسافات صحيحة بين الكلمات. "
    "الهدف من هذا الاختبار هو التأكد من أن النص المستخرج سليم وقابل للاستخدام في توليد بيانات التدريب."
) * 3


def test_clean_arabic_text_passes():
    result = validate_extraction(CLEAN_ARABIC)
    assert result.passed
    assert result.hard_failures == []
    assert result.confidence > 0.8


def test_front_matter_page_with_leading_roman_numeral_fails():
    text = "iii\n" + CLEAN_ARABIC
    result = validate_extraction(text)
    assert not result.passed
    assert "front_matter_page" in result.hard_failures


def test_front_matter_roman_numeral_detected_within_first_few_lines():
    text = "8 المستوى\n1 الجزء\nv\n" + CLEAN_ARABIC
    result = validate_extraction(text)
    assert "front_matter_page" in result.hard_failures


def test_roman_numeral_word_deep_in_body_text_does_not_trigger_front_matter():
    text = "\n".join([CLEAN_ARABIC] * 7) + "\niii\n" + CLEAN_ARABIC
    result = validate_extraction(text)
    assert "front_matter_page" not in result.hard_failures


def test_front_matter_roman_numerals_above_thirty_nine_are_detected():
    for numeral in ("xl", "xlii", "xlix", "l", "li", "lxxxviii", "xcix"):
        text = f"{numeral}\n" + CLEAN_ARABIC
        result = validate_extraction(text)
        assert "front_matter_page" in result.hard_failures, f"{numeral} should be detected"


def test_rubric_boilerplate_page_with_multiple_scale_terms_fails():
    text = CLEAN_ARABIC + "\nدون المعيار\nيقترب من المعيار\nضمن المعيار\nيفوق المعيار\n"
    result = validate_extraction(text)
    assert not result.passed
    assert "rubric_boilerplate_page" in result.hard_failures


def test_single_incidental_mention_of_meaayir_does_not_trigger_rubric_exclusion():
    text = CLEAN_ARABIC + "\nمؤشّرات التعلّم: المعايير الأساسية لهذه الوحدة.\n"
    result = validate_extraction(text)
    assert "rubric_boilerplate_page" not in result.hard_failures


def test_too_short_content_fails():
    result = validate_extraction("نص قصير")
    assert not result.passed
    assert "too_short" in result.hard_failures
    assert result.confidence == 0.0


def test_english_only_content_fails_arabic_ratio_check():
    english_text = "This is a plain English paragraph with no Arabic content at all. " * 3
    result = validate_extraction(english_text, expect_arabic=True)
    assert not result.passed
    assert "low_arabic_content" in result.hard_failures


def test_english_only_content_passes_when_arabic_not_expected():
    english_text = "This is a plain English paragraph with no Arabic content at all. " * 3
    result = validate_extraction(english_text, expect_arabic=False)
    assert result.passed


def test_broken_text_layer_with_presentation_forms_fails():
    presentation_forms = "".join(chr(cp) for cp in range(0xFE70, 0xFEFF, 2))
    broken_text = (presentation_forms * 10) + " " + CLEAN_ARABIC
    result = validate_extraction(broken_text)
    assert not result.passed
    assert "broken_text_layer_needs_ocr" in result.hard_failures


def test_encoding_errors_from_replacement_chars_fail():
    corrupted = CLEAN_ARABIC[:50] + ("�" * 20) + CLEAN_ARABIC[50:]
    result = validate_extraction(corrupted)
    assert not result.passed
    assert "encoding_errors" in result.hard_failures


def test_excessive_repetition_is_soft_warning_not_hard_failure():
    repeated_line = "هذا سطر متكرر يظهر في كل صفحة من صفحات الكتاب المدرسي.\n"
    text = repeated_line * 20
    result = validate_extraction(text)
    assert result.passed
    assert "excessive_repetition" in result.soft_warnings
    assert result.confidence < 1.0


def test_no_spaces_triggers_abnormal_word_length_warning():
    no_space_text = "أ" * 500
    result = validate_extraction(no_space_text)
    assert "abnormal_word_length_broken_tokenization" in result.soft_warnings


def test_metrics_are_reported():
    result = validate_extraction(CLEAN_ARABIC)
    assert "arabic_ratio" in result.metrics
    assert "length" in result.metrics
    assert result.metrics["length"] == len(CLEAN_ARABIC)
