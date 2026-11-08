// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # French Curated Synonym Table
//!
//! Compile-time perfect-hash map for French vocabulary.  Used by
//! [`super::SynonymBank`] when [`super::detect::LanguageHint::French`] is active.

use phf::{Map, phf_map};

/// Curated French synonym table.
///
/// Applied when [`super::detect::LanguageHint::French`] is selected.
///
/// # Lookup Characteristics
///
/// Lookups are O(1) and require no heap allocation.
pub(super) static FRENCH_SYNONYMS: Map<&'static str, &'static [&'static str]> = phf_map! {
    "commence"    => &["débute", "démarre", "entame", "amorce", "ouvre"],
    "contient"    => &["inclut", "renferme", "comporte", "englobe", "abrite"],
    "supprime"    => &["efface", "retire", "élimine", "ôte", "enlève"],
    "texte"       => &["contenu", "prose", "rédaction", "écrit", "matière"],
    "format"      => &["structure", "agencement", "forme", "style", "schéma"],
    "mot"         => &["terme", "vocable", "expression", "lexème", "unité"],
    "invisible"   => &["caché", "imperceptible", "dissimulé", "latent", "discret"],
    "marque"      => &["signal", "indicateur", "empreinte", "étiquette", "repère"],
    "important"   => &["significatif", "essentiel", "crucial", "fondamental", "capital"],
    "permet"      => &["autorise", "habilite", "laisse", "facilite"],
    "génère"      => &["produit", "crée", "engendre", "élabore", "construit"],
    "définit"     => &["spécifie", "établit", "détermine", "fixe", "délimite"],
    "traite"      => &["gère", "opère", "administre", "examine", "analyse"],
    "système"     => &["plateforme", "architecture", "mécanisme", "cadre"],
    "processus"   => &["procédure", "méthode", "opération", "flux", "pipeline"],
    "résultat"    => &["sortie", "produit", "rendement", "conséquence"],
    "analyse"     => &["examen", "étude", "évaluation", "révision", "inspection"],
    "données"     => &["information", "registres", "métriques", "input", "faits"],
    "nouveau"     => &["novateur", "moderne", "récent", "inédit", "frais"],
    "grand"       => &["vaste", "étendu", "considérable", "énorme", "important"],
    "petit"       => &["minuscule", "réduit", "bref", "compact", "minimal"],
    "rapide"      => &["vif", "alerte", "prompt", "expéditif", "agile"],
    "simple"      => &["élémentaire", "basique", "facile", "direct", "clair"],
    "complexe"    => &["élaboré", "sophistiqué", "intriqué", "profond"],
    "toujours"    => &["invariablement", "constamment", "perpétuellement"],
    "jamais"      => &["en aucun cas", "nullement"],
    "également"   => &["aussi", "pareillement", "autant"],
};
