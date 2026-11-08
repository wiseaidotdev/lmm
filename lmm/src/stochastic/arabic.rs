// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Arabic Curated Synonym Table
//!
//! Compile-time perfect-hash map for Arabic vocabulary.  Used by
//! [`super::SynonymBank`] when [`super::detect::LanguageHint::Arabic`] is active.

use phf::{Map, phf_map};

/// Curated Arabic synonym table.
///
/// Maps transliterated Arabic head words and Arabic-script tokens to
/// semantically equivalent Arabic-script alternatives.  Applied when
/// [`super::detect::LanguageHint::Arabic`] is selected.
///
/// # Lookup Characteristics
///
/// Lookups are O(1) and require no heap allocation (perfect hash at compile
/// time via the `phf` crate).
pub(super) static ARABIC_SYNONYMS: Map<&'static str, &'static [&'static str]> = phf_map! {
    "يبدأ"      => &["يشرع", "يفتتح", "يطلق", "يستهل", "يمضي"],
    "يحتوي"     => &["يتضمن", "يشمل", "يضم", "يحمل", "يستوعب"],
    "نص"        => &["محتوى", "كتابة", "مقال", "مادة", "فقرة"],
    "مهم"       => &["أساسي", "جوهري", "بالغ", "محوري", "ضروري"],
    "يولد"      => &["ينتج", "يصنع", "يبني", "يُنشئ", "يُخرج"],
    "يعرف"      => &["يحدد", "يضبط", "يرسم", "يصف", "يُعيّن"],
    "نظام"      => &["منظومة", "هيكل", "بنية", "منهج", "إطار"],
    "نتيجة"     => &["مخرج", "محصلة", "ثمرة", "عائد", "حصيلة"],
    "تحليل"     => &["دراسة", "فحص", "تمحيص", "مراجعة", "استقراء"],
    "بيانات"    => &["معلومات", "إحصاءات", "وقائع", "أرقام", "سجلات"],
    "جديد"      => &["مستحدث", "حديث", "طازج", "مبتكر", "معاصر"],
    "كبير"      => &["ضخم", "واسع", "هائل", "شاسع", "عريض"],
    "صغير"      => &["ضئيل", "محدود", "طفيف", "مصغّر", "مقتضب"],
    "سريع"      => &["خاطف", "عاجل", "متسارع", "وثّاب", "رشيق"],
    "بطيء"      => &["متأنٍّ", "متمهل", "هادئ", "رزين", "بطيء"],
    "بسيط"      => &["سهل", "مبسّط", "واضح", "مباشر", "ميسّر"],
    "معقد"      => &["متشعب", "متداخل", "عسير", "دقيق", "مركّب"],
    "دائما"     => &["باستمرار", "على الدوام", "أبدا", "دوما", "سرمديا"],
    "أبدا"      => &["قط", "البتة", "لا مطلقا", "في حال من الأحوال"],
    "أيضا"      => &["كذلك", "علاوة على ذلك", "فضلا", "بالمثل"],
    "لأن"       => &["إذ", "نظرا لأن", "بما أن", "بسبب أن", "حيث إن"],
    "رغم"       => &["على الرغم من", "بالرغم من", "في حين أن"],
    "علاوة"     => &["فضلا", "إضافة", "كذلك", "إلى ذلك"],
};
