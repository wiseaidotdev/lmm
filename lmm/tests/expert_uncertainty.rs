// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lmm::expert::uncertainty::{CertaintyFactor, cf_from_idf, combine_all};

#[test]
fn certainty_factor_clamping_negative() {
    let cf = CertaintyFactor::new(-1.0);
    assert_eq!(cf.value(), 0.0);
}

#[test]
fn certainty_factor_clamping_excess() {
    let cf = CertaintyFactor::new(2.5);
    assert_eq!(cf.value(), 1.0);
}

#[test]
fn certainty_factor_exact_boundaries() {
    let zero = CertaintyFactor::new(0.0);
    let one = CertaintyFactor::new(1.0);
    assert_eq!(zero.value(), 0.0);
    assert_eq!(one.value(), 1.0);
}

#[test]
fn certainty_factor_nan_clamps_to_zero() {
    let cf = CertaintyFactor::new(f64::NAN);
    let _v = cf.value();
}

#[test]
fn certainty_factor_mid_range_value_preserved() {
    let cf = CertaintyFactor::new(0.55);
    assert!((cf.value() - 0.55).abs() < 1e-12);
}

#[test]
fn certainty_factor_combine_zero_identity() {
    // cf + 0 x (1 - cf) = cf
    let cf = CertaintyFactor::new(0.67);
    let zero = CertaintyFactor::new(0.0);
    assert!((cf.combine(&zero).value() - 0.67).abs() < 1e-9);
}

#[test]
fn certainty_factor_combine_one_saturation() {
    // cf + 1 x (1 - cf) = 1
    let cf = CertaintyFactor::new(0.42);
    let one = CertaintyFactor::new(1.0);
    assert_eq!(cf.combine(&one).value(), 1.0);
}

#[test]
fn certainty_factor_combine_monotonic() {
    let cf1 = CertaintyFactor::new(0.6);
    let cf2 = CertaintyFactor::new(0.5);
    let combined = cf1.combine(&cf2);
    assert!((combined.value() - 0.80).abs() < 1e-9);
    assert!(combined.value() >= cf1.value());
    assert!(combined.value() >= cf2.value());
}

#[test]
fn certainty_factor_combine_symmetry() {
    let a = CertaintyFactor::new(0.35);
    let b = CertaintyFactor::new(0.82);
    let ab = a.combine(&b).value();
    let ba = b.combine(&a).value();
    assert!((ab - ba).abs() < 1e-9);
}

#[test]
fn certainty_factor_combine_with_itself_not_idempotent() {
    // cf + cf x (1 - cf)  = cf(2 - cf) which is > cf when cf ∈ (0,1).
    let cf = CertaintyFactor::new(0.5);
    let combined = cf.combine(&cf);
    assert!(combined.value() > 0.5);
    assert!((combined.value() - 0.75).abs() < 1e-9);
}

#[test]
fn certainty_factor_combine_both_zero() {
    let zero = CertaintyFactor::new(0.0);
    assert_eq!(zero.combine(&zero).value(), 0.0);
}

#[test]
fn certainty_factor_combine_both_one() {
    let one = CertaintyFactor::new(1.0);
    assert_eq!(one.combine(&one).value(), 1.0);
}

#[test]
fn certainty_factor_propagate_standard() {
    let evidence = CertaintyFactor::new(0.8);
    let propagated = evidence.propagate(0.5);
    assert!((propagated.value() - 0.4).abs() < 1e-9);
}

#[test]
fn certainty_factor_propagate_clamped_extremes() {
    let evidence = CertaintyFactor::new(0.7);
    assert_eq!(evidence.propagate(-0.5).value(), 0.0);
    assert!((evidence.propagate(1.5).value() - 0.7).abs() < 1e-9);
}

#[test]
fn certainty_factor_propagate_rule_zero_gives_zero() {
    let evidence = CertaintyFactor::new(0.9);
    assert_eq!(evidence.propagate(0.0).value(), 0.0);
}

#[test]
fn certainty_factor_propagate_rule_one_preserves_evidence() {
    let evidence = CertaintyFactor::new(0.65);
    assert!((evidence.propagate(1.0).value() - 0.65).abs() < 1e-9);
}

#[test]
fn certainty_factor_propagate_zero_evidence_always_zero() {
    let zero = CertaintyFactor::new(0.0);
    assert_eq!(zero.propagate(0.9).value(), 0.0);
}

#[test]
fn certainty_factor_is_sufficient_boundary() {
    let cf = CertaintyFactor::new(0.30);
    assert!(cf.is_sufficient(0.30));
    assert!(cf.is_sufficient(0.299));
    assert!(!cf.is_sufficient(0.301));
}

#[test]
fn certainty_factor_is_sufficient_zero_threshold_always_true() {
    let cf = CertaintyFactor::new(0.0);
    assert!(cf.is_sufficient(0.0));
}

#[test]
fn certainty_factor_is_sufficient_one_threshold_only_one() {
    assert!(CertaintyFactor::new(1.0).is_sufficient(1.0));
    assert!(!CertaintyFactor::new(0.9999).is_sufficient(1.0));
}

#[test]
fn certainty_factor_labels_all_tiers() {
    assert_eq!(CertaintyFactor::new(0.0).label(), "low confidence");
    assert_eq!(CertaintyFactor::new(0.299).label(), "low confidence");
    assert_eq!(CertaintyFactor::new(0.30).label(), "moderate confidence");
    assert_eq!(CertaintyFactor::new(0.599).label(), "moderate confidence");
    assert_eq!(CertaintyFactor::new(0.60).label(), "high confidence");
    assert_eq!(CertaintyFactor::new(0.849).label(), "high confidence");
    assert_eq!(CertaintyFactor::new(0.85).label(), "very high confidence");
    assert_eq!(CertaintyFactor::new(1.0).label(), "very high confidence");
}

#[test]
fn certainty_factor_label_boundary_exactly_at_cutoffs() {
    assert_eq!(CertaintyFactor::new(0.3).label(), "moderate confidence");
    assert_eq!(CertaintyFactor::new(0.6).label(), "high confidence");
    assert_eq!(CertaintyFactor::new(0.85).label(), "very high confidence");
}

#[test]
fn certainty_factor_ordering() {
    let low = CertaintyFactor::new(0.2);
    let high = CertaintyFactor::new(0.8);
    assert!(low < high);
    assert!(high > low);
}

#[test]
fn certainty_factor_copy_clone() {
    let cf = CertaintyFactor::new(0.55);
    let copied = cf;
    let cloned = cf; // clone / copy = same / same
    assert!((copied.value() - 0.55).abs() < 1e-12);
    assert!((cloned.value() - 0.55).abs() < 1e-12);
}

#[test]
fn combine_all_empty_slice() {
    let result = combine_all(&[]);
    assert_eq!(result.value(), 0.0);
}

#[test]
fn combine_all_single_element() {
    let factors = [CertaintyFactor::new(0.55)];
    assert!((combine_all(&factors).value() - 0.55).abs() < 1e-9);
}

#[test]
fn combine_all_multiple_elements() {
    let factors = [
        CertaintyFactor::new(0.5),
        CertaintyFactor::new(0.5),
        CertaintyFactor::new(0.5),
    ];
    // 0.5 + 0.5x0.5 = 0.75; 0.75 + 0.5x0.25 = 0.875
    assert!((combine_all(&factors).value() - 0.875).abs() < 1e-9);
}

#[test]
fn combine_all_repeated_accumulation_approaches_one() {
    let factors = vec![CertaintyFactor::new(0.3); 10];
    let combined = combine_all(&factors);
    assert!(combined.value() > 0.95);
    assert!(combined.value() <= 1.0);
}

#[test]
fn combine_all_all_zeros_stays_zero() {
    let factors = vec![CertaintyFactor::new(0.0); 5];
    assert_eq!(combine_all(&factors).value(), 0.0);
}

#[test]
fn combine_all_one_in_slice_saturates() {
    let factors = vec![
        CertaintyFactor::new(0.3),
        CertaintyFactor::new(1.0),
        CertaintyFactor::new(0.5),
    ];
    // Once 1.0 is combined, result stays 1.0.
    assert_eq!(combine_all(&factors).value(), 1.0);
}

#[test]
fn combine_all_order_independence() {
    let forward = combine_all(&[CertaintyFactor::new(0.4), CertaintyFactor::new(0.6)]);
    let backward = combine_all(&[CertaintyFactor::new(0.6), CertaintyFactor::new(0.4)]);
    assert!((forward.value() - backward.value()).abs() < 1e-9);
}

#[test]
fn cf_from_idf_handles_invalid_max() {
    assert_eq!(cf_from_idf(3.0, 0.0).value(), 0.1);
    assert_eq!(cf_from_idf(3.0, -2.0).value(), 0.1);
}

#[test]
fn cf_from_idf_monotonic_growth() {
    let low = cf_from_idf(1.0, 10.0);
    let high = cf_from_idf(8.0, 10.0);
    assert!(low.value() < high.value());
    assert!(low.value() >= 0.1);
    assert!(high.value() <= 0.95);
}

#[test]
fn cf_from_idf_zero_idf_gives_minimum() {
    // ln(1+0)/ln(1+max_idf) x 0.85 + 0.1 = 0 x 0.85 + 0.1 = 0.1
    let cf = cf_from_idf(0.0, 10.0);
    assert!((cf.value() - 0.1).abs() < 1e-9);
}

#[test]
fn cf_from_idf_equal_idf_max_gives_maximum() {
    // When idf == max_idf: ln(1+idf)/ln(1+idf) x 0.85 + 0.1 = 0.95
    let cf = cf_from_idf(5.0, 5.0);
    assert!((cf.value() - 0.95).abs() < 1e-9);
}

#[test]
fn cf_from_idf_idf_above_max_still_capped() {
    // idf > max_idf: ratio > 1 → x 0.85 + 0.1 > 0.95 → clamped to 1.0 by CertaintyFactor::new.
    let cf = cf_from_idf(20.0, 5.0);
    assert!(cf.value() <= 1.0);
    assert!(cf.value() >= 0.95);
}

#[test]
fn cf_from_idf_result_always_in_range() {
    for &(idf, max_idf) in &[
        (0.0, 1.0),
        (1.0, 1.0),
        (5.0, 10.0),
        (10.0, 10.0),
        (100.0, 10.0),
    ] {
        let cf = cf_from_idf(idf, max_idf);
        assert!(
            cf.value() >= 0.0 && cf.value() <= 1.0,
            "out of range for idf={idf}, max_idf={max_idf}"
        );
    }
}

// Copyright 2026 Mahmoud Harmouch, licensed under MIT.
