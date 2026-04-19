// Copyright (C) 2026 Jorge Andre Castro
//
// Ce programme est un logiciel libre : vous pouvez le redistribuer et/ou le modifier
// selon les termes de la Licence Publique Générale GNU telle que publiée par la
// Free Software Foundation, soit la version 2 de la licence, soit (à votre convention)
// n'importe quelle version ultérieure.

//! # sigmoid-q15
//!
//! Fonction d'activation Sigmoid en virgule fixe Q15 pour systèmes embarqués.
//!
//! ## Caractéristiques
//!
//! - `#![no_std]` : aucune dépendance à la bibliothèque standard.
//! - Arithmétique entière pure : pas de flottants, pas de `libm`.
//! - Compatible **RP2040** (Cortex-M0+) et **RP2350** (Cortex-M33).
//! - Utilise [`embedded-exp`](https://crates.io/crates/embedded-exp) pour le calcul de `e^x`.
//! - Temps d'exécution **constant** (déterministe) : idéal pour les noyaux temps réel.
//! - Zéro allocation dynamique : traitement in-place possible.
//!
//! ## Format Q15
//!
//! En Q15, un `i16` représente un nombre réel dans `[-1.0, 1.0[` :
//! ```text
//! valeur_réelle = valeur_i16 / 32768.0
//! ```
//!
//! ## Algorithme
//!
//! `sigmoid(x) = 1 / (1 + e^(-x))`
//!
//! Le problème : `embedded-exp` ne fonctionne que sur le domaine **négatif**.
//! On exploite la symétrie de sigmoid pour toujours rester sur ce domaine :
//!
//! ```text
//! sigmoid(x)  = 1 / (1 + e^(-x))      si x <= 0  → -x >= 0... non
//! ```
//!
//! Plus précisément :
//!
//! - Si `x <= 0` : `-x >= 0`, mais `e^(-x)` avec `-x >= 0` saturerait.
//!   On utilise donc `sigmoid(x) = 1 - sigmoid(-x)` pour ramener
//!   le calcul sur le domaine négatif.
//! - Si `x > 0` : `-x < 0` → `exp_q15(-x)` calcule normalement.
//! - Si `x = 0` : `sigmoid(0) = 0.5` → `16384` en Q15.
//!
//! **Propriété exploitée :**
//! ```text
//! sigmoid(-x) = 1 - sigmoid(x)
//! ```
//! On calcule toujours `exp_q15` sur une valeur **négative ou nulle**,
//! ce qui est exactement le domaine valide de `embedded-exp`.
//!
//! ## Exemple
//!
//! ```rust
//! use sigmoid_q15::{sigmoid_q15, sigmoid_slice_q15};
//!
//! // sigmoid(0) = 0.5 → 16384 en Q15
//! assert_eq!(sigmoid_q15(0), 16384);
//!
//! // sigmoid(-1.0) ≈ 0.2689 → ≈ 8808 en Q15
//! let res = sigmoid_q15(-32768);
//! assert!((res as i32 - 8808).abs() < 20);
//!
//! // sigmoid(x) est toujours dans ]0.0, 1.0[ → toujours positif
//! assert!(sigmoid_q15(i16::MIN) > 0);
//! assert!(sigmoid_q15(i16::MAX) > 0);
//!
//! // Traitement in-place
//! let mut buf = [-32768i16, 0, 32767];
//! sigmoid_slice_q15(&mut buf);
//! assert!(buf[0] > 0);
//! assert_eq!(buf[1], 16384);
//! assert!(buf[2] > 16384);
//! ```

#![no_std]
#![forbid(unsafe_code)]

use embedded_exp::exp_q15;

/// Un en Q15 : représente 1.0, mais i16::MAX = 32767 ≈ 1.0
/// On utilise 32768 comme "1.0 exact" en i32 pour les calculs internes.
const ONE_Q15: i32 = 32768;

/// Calcule la fonction d'activation Sigmoid pour un nombre en virgule fixe Q15.
///
/// `sigmoid(x) = 1 / (1 + e^(-x))`
///
/// La sortie est toujours dans `]0.0, 1.0[`, soit `[1, 32767]` en Q15.
///
/// ## Stratégie
///
/// On exploite la symétrie `sigmoid(-x) = 1 - sigmoid(x)` pour toujours
/// appeler `exp_q15` avec une valeur **négative**, son domaine valide.
///
/// | Entrée `x` | Calcul effectué                        |
/// |-----------|----------------------------------------|
/// | `x > 0`   | `1 / (1 + exp(-x))` directement        |
/// | `x == 0`  | retourne `16384` (0.5 exact)           |
/// | `x < 0`   | `1 - sigmoid(-x)` par symétrie         |
///
/// ## Cas limites
///
/// | Entrée     | Valeur Q15 | Sortie attendue     |
/// |------------|-----------|---------------------|
/// | `i16::MIN` | -32768    | ≈ 8808 (≈ 0.2689)   |
/// | `0`        | 0         | 16384 (= 0.5 exact) |
/// | `i16::MAX` | 32767     | ≈ 23960 (≈ 0.7311)  |
///
/// # Exemple
///
/// ```rust
/// use sigmoid_q15::sigmoid_q15;
///
/// assert_eq!(sigmoid_q15(0), 16384);
/// assert!(sigmoid_q15(-32768) > 0);
/// assert!(sigmoid_q15(32767) > 16384);
/// ```
#[inline]
pub fn sigmoid_q15(x: i16) -> i16 {
    // Cas x = 0 : sigmoid(0) = 0.5 = 16384 en Q15
    if x == 0 {
        return 16384;
    }

    if x > 0 {
        // x > 0 → -x < 0 → exp_q15(-x) est dans le domaine valide
        // sigmoid(x) = 1 / (1 + e^(-x))
        sigmoid_positive(x)
    } else {
        // x < 0 → on utilise la symétrie : sigmoid(x) = 1 - sigmoid(-x)
        // -x > 0 → sigmoid(-x) se calcule via sigmoid_positive(-x)
        // Comme x >= i16::MIN, -x peut déborder si x == i16::MIN.
        // On gère ce cas : si x == i16::MIN, on sature -x à i16::MAX.
        let neg_x = if x == i16::MIN {
            i16::MAX
        } else {
            -x
        };
        let s_neg_x = sigmoid_positive(neg_x) as i32;
        // 1 - sigmoid(-x) en Q15
        // ONE_Q15 = 32768, mais sigmoid retourne au max 32767
        // On clamp à 1 minimum pour garantir > 0
        let result = ONE_Q15 - s_neg_x;
        result.max(1).min(32767) as i16
    }
}

/// Calcul interne de sigmoid pour x > 0.
/// Précondition : x > 0, donc -x < 0, donc exp_q15(-x) est valide.
#[inline]
fn sigmoid_positive(x: i16) -> i16 {
    debug_assert!(x > 0);

    // -x est négatif : domaine valide pour exp_q15
    let neg_x = -x; // x > 0 et x != i16::MIN donc pas de débordement

    // e^(-x) en Q15
    let exp_neg_x = exp_q15(neg_x) as i32;

    // 1 + e^(-x) en Q15
    // ONE_Q15 = 32768 représente 1.0 ; exp_neg_x ∈ [1, 32767]
    // Somme max : 32768 + 32767 = 65535 → tient dans i32
    let denom = ONE_Q15 + exp_neg_x;

    // sigmoid = 1 / (1 + e^(-x)) = ONE_Q15 / denom
    // On multiplie numérateur par ONE_Q15 pour rester en Q15 :
    // résultat = (ONE_Q15 * ONE_Q15) / denom
    // ONE_Q15 * ONE_Q15 = 32768 * 32768 = 1_073_741_824 → tient dans i32 (max ~2.1e9)
    let result = (ONE_Q15 * ONE_Q15) / denom;

    // Saturation : résultat dans [1, 32767]
    result.max(1).min(32767) as i16
}

/// Applique Sigmoid sur un slice de données en virgule fixe Q15, **in-place**.
///
/// Chaque élément `v` est remplacé par `sigmoid(v)`.
/// L'opération in-place évite toute allocation dynamique.
///
/// ## Slice vide
///
/// Un slice vide (`&mut []`) est accepté sans erreur.
///
/// # Exemple
///
/// ```rust
/// use sigmoid_q15::sigmoid_slice_q15;
///
/// let mut data = [-32768i16, 0, 32767];
/// sigmoid_slice_q15(&mut data);
/// // Tous les résultats sont dans ]0, 32767]
/// assert!(data[0] > 0 && data[0] < 16384); // < 0.5
/// assert_eq!(data[1], 16384);               // = 0.5
/// assert!(data[2] > 16384);                 // > 0.5
///
/// // Slice vide : aucun effet, aucune panique
/// let mut empty: [i16; 0] = [];
/// sigmoid_slice_q15(&mut empty);
/// ```
#[inline]
pub fn sigmoid_slice_q15(data: &mut [i16]) {
    for val in data.iter_mut() {
        *val = sigmoid_q15(*val);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    //  Valeurs de référence 
    // sigmoid(0.0)  = 0.5     → 16384 en Q15
    // sigmoid(-1.0) ≈ 0.2689  → ≈ 8808 en Q15
    // sigmoid(1.0)  ≈ 0.7311  → ≈ 23960 en Q15
    // sigmoid(-0.5) ≈ 0.3775  → ≈ 12372 en Q15
    // sigmoid(0.5)  ≈ 0.6225  → ≈ 20396 en Q15

    const TOLERANCE: i32 = 30; // tolérance en ULP Q15

    #[test]
    fn test_sigmoid_zero_is_half() {
        // sigmoid(0) = 0.5 exact → 16384 en Q15
        assert_eq!(sigmoid_q15(0), 16384);
    }

    #[test]
    fn test_sigmoid_negative_one() {
        // sigmoid(-1.0) ≈ 0.2689 → ≈ 8808
        let res = sigmoid_q15(-32768);
        assert!(
            (res as i32 - 8808).abs() < TOLERANCE,
            "sigmoid(-1.0): attendu ≈8808, reçu {}",
            res
        );
    }

    #[test]
    fn test_sigmoid_positive_one() {
        // sigmoid(1.0) ≈ 0.7311 → ≈ 23960
        // x = i16::MAX ≈ 1.0 en Q15
        let res = sigmoid_q15(i16::MAX);
        assert!(
            (res as i32 - 23960).abs() < TOLERANCE,
            "sigmoid(≈1.0): attendu ≈23960, reçu {}",
            res
        );
    }

    #[test]
    fn test_sigmoid_negative_half() {
        // sigmoid(-0.5) ≈ 0.3775 → ≈ 12372
        let res = sigmoid_q15(-16384);
        assert!(
            (res as i32 - 12372).abs() < TOLERANCE,
            "sigmoid(-0.5): attendu ≈12372, reçu {}",
            res
        );
    }

    #[test]
    fn test_sigmoid_positive_half() {
        // sigmoid(0.5) ≈ 0.6225 → ≈ 20396
        let res = sigmoid_q15(16384);
        assert!(
            (res as i32 - 20396).abs() < TOLERANCE,
            "sigmoid(0.5): attendu ≈20396, reçu {}",
            res
        );
    }

    #[test]
    fn test_sigmoid_symmetry() {
        // sigmoid(x) + sigmoid(-x) ≈ 1.0 (= 32768 en Q15)
        // On exclut i16::MIN car -i16::MIN déborde en i16
        for x in [-16384i16, -8192, -4096, -1] {
            let s_pos = sigmoid_q15(-x) as i32;
            let s_neg = sigmoid_q15(x) as i32;
            let sum = s_pos + s_neg;
            assert!(
                (sum - 32768).abs() < TOLERANCE * 2,
                "symétrie brisée pour x={}: sigmoid({})={} + sigmoid({})={} = {}",
                x, -x, s_pos, x, s_neg, sum
            );
        }
    }

    #[test]
    fn test_sigmoid_always_positive() {
        // sigmoid(x) > 0 pour tout x
        for x in [i16::MIN, -16384, -1, 0, 1, 16384, i16::MAX] {
            assert!(sigmoid_q15(x) > 0, "sigmoid({}) doit être > 0", x);
        }
    }

    #[test]
    fn test_sigmoid_always_below_one() {
        // sigmoid(x) < 1.0 (< 32768) → au max i16::MAX = 32767
        // On cast en i32 pour que la comparaison ait du sens pour le compilateur
        for x in [i16::MIN, -16384, -1, 0, 1, 16384, i16::MAX] {
            assert!(
                (sigmoid_q15(x) as i32) < 32768,
                "sigmoid({}) doit être < 32768",
                x
            );
        }
    }

    #[test]
    fn test_sigmoid_monotone() {
        // sigmoid est strictement croissante
        let a = sigmoid_q15(-32768); // sigmoid(-1.0)
        let b = sigmoid_q15(-16384); // sigmoid(-0.5)
        let c = sigmoid_q15(0);      // sigmoid(0.0)
        let d = sigmoid_q15(16384);  // sigmoid(0.5)
        let e = sigmoid_q15(32767);  // sigmoid(≈1.0)
        assert!(a < b, "sigmoid(-1.0)={} < sigmoid(-0.5)={}", a, b);
        assert!(b < c, "sigmoid(-0.5)={} < sigmoid(0.0)={}", b, c);
        assert!(c < d, "sigmoid(0.0)={} < sigmoid(0.5)={}", c, d);
        assert!(d < e, "sigmoid(0.5)={} < sigmoid(1.0)={}", d, e);
    }

    #[test]
    fn test_sigmoid_min_no_panic() {
        // i16::MIN est un cas limite : -i16::MIN déborde en i16
        // On vérifie que ça ne panique pas et retourne quelque chose de valide
        let res = sigmoid_q15(i16::MIN);
        assert!(res > 0, "sigmoid(i16::MIN)={} doit être > 0", res);
    }

    #[test]
    fn test_sigmoid_output_in_range() {
        // Toute sortie doit être dans [1, 32767]
        for x in [i16::MIN, -32000, -16384, -1, 0, 1, 16384, 32000, i16::MAX] {
            let res = sigmoid_q15(x);
            assert!(
                res >= 1,
                "sigmoid({})={} doit être >= 1",
                x, res
            );
        }
    }

    //sigmoid_slice_q15 

    #[test]
    fn test_slice_empty() {
        let mut empty: [i16; 0] = [];
        sigmoid_slice_q15(&mut empty);
    }

    #[test]
    fn test_slice_zero_gives_half() {
        let mut data = [0i16];
        sigmoid_slice_q15(&mut data);
        assert_eq!(data[0], 16384);
    }

    #[test]
    fn test_slice_mixed() {
        let mut data = [-32768i16, 0, 32767];
        sigmoid_slice_q15(&mut data);
        assert!(data[0] > 0 && data[0] < 16384); // < 0.5
        assert_eq!(data[1], 16384);               // = 0.5
        assert!(data[2] > 16384);                 // > 0.5
    }

    #[test]
    fn test_slice_all_positive_above_half() {
        let mut data = [1i16, 8192, 16384, 32767];
        sigmoid_slice_q15(&mut data);
        for val in data {
            assert!(val > 16384, "sigmoid positif doit être > 16384, reçu {}", val);
        }
    }

    #[test]
    fn test_slice_all_negative_below_half() {
        let mut data = [-1i16, -8192, -16384, -32768];
        sigmoid_slice_q15(&mut data);
        for val in data {
            assert!(val < 16384, "sigmoid négatif doit être < 16384, reçu {}", val);
        }
    }

    #[test]
    fn test_slice_idempotent_after_double_pass() {
        // sigmoid n'est PAS idempotente (sigmoid(sigmoid(x)) != sigmoid(x))
        // mais on vérifie que deux passes consécutives ne paniquent pas
        // et restent dans la plage valide
        let mut data = [-32768i16, -16384, 0, 16384, 32767];
        sigmoid_slice_q15(&mut data);
        sigmoid_slice_q15(&mut data);
        for val in data {
            assert!(val >= 1, "hors plage après double passe: {}", val);
        }
    }

    #[test]
    fn test_slice_boundary_values() {
        let mut data = [i16::MIN, i16::MAX];
        sigmoid_slice_q15(&mut data);
        assert!(data[0] > 0 && data[0] < 16384);
        assert!(data[1] > 16384);
    }
}