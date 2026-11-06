



use std::sync::Mutex;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use regex::Regex;
use rustc_hash::FxHashMap as HashMap;

const BOS: u16 = 257;
const EOS: u16 = 258;
const FIRST_MERGE: u32 = 261;

const SPACE: &str = r"\t\n\x0B\x0C\r\x1C-\x1F \x{85}\x{A0}\x{1680}\x{2000}-\x{200A}\x{2028}\x{2029}\x{202F}\x{205F}\x{3000}";

const LATIN: &str = r"A-Za-z\x{C0}-\x{D6}\x{D8}-\x{F6}\x{F8}-\x{24F}\x{1E00}-\x{1EFF}";
const HANGUL: &str = r"\x{AC00}-\x{D7A3}\x{1100}-\x{11FF}\x{3130}-\x{318F}";
const KANA: &str = r"\x{3040}-\x{309F}\x{30A0}-\x{30FF}\x{31F0}-\x{31FF}";
const HAN: &str = r"\x{3400}-\x{4DBF}\x{4E00}-\x{9FFF}\x{F900}-\x{FAFF}";
const CACHE_LIMIT: usize = 4_000_000;

fn pattern(version: u32) -> Option<String> {
    match version {
        2 => Some(format!(r"```[A-Za-z]*| ?[A-Za-z]+| ?[0-9]+|[{SPACE}]+|[^A-Za-z0-9{SPACE}]")),
        3 => Some(format!(r"```[A-Za-z]*| ?[{LATIN}]+| ?[{HANGUL}]+|[{KANA}]+|[{HAN}]+| ?[0-9]+|[{SPACE}]+|[^{SPACE}]")),
        _ => None,
    }
}

#[pyclass(frozen)]
struct Tokenizer {
    pattern: Regex,
    ranks: HashMap<(u16, u16), (u32, u16)>,
    cache: Mutex<HashMap<String, Vec<u16>>>,
}

impl Tokenizer {

    fn merge(&self, piece: &[u8]) -> Vec<u16> {
        let mut sequence: Vec<u16> = piece.iter().map(|&b| b as u16).collect();
        while sequence.len() > 1 {
            let best = sequence.windows(2)
                .filter_map(|w| self.ranks.get(&(w[0], w[1])).map(|&(rank, token)| (rank, token, w[0], w[1])))
                .min();
            let Some((_, token, a, b)) = best else { break };
            let mut merged = Vec::with_capacity(sequence.len());
            let mut i = 0;
            while i < sequence.len() {
                if i + 1 < sequence.len() && sequence[i] == a && sequence[i + 1] == b {
                    merged.push(token);
                    i += 2;
                } else {
                    merged.push(sequence[i]);
                    i += 1;
                }
            }
            sequence = merged;
        }
        sequence
    }

    fn extend(&self, text: &str, out: &mut Vec<u16>) {
        let mut cache = self.cache.lock().unwrap();
        for found in self.pattern.find_iter(text) {
            let piece = found.as_str();
            if let Some(tokens) = cache.get(piece) {
                out.extend_from_slice(tokens);
                continue;
            }
            let tokens = self.merge(piece.as_bytes());
            out.extend_from_slice(&tokens);
            if cache.len() >= CACHE_LIMIT {
                cache.clear();
            }
            cache.insert(piece.to_owned(), tokens);
        }
    }
}

#[pymethods]
impl Tokenizer {
    #[new]
    #[pyo3(signature = (merges, version=2))]
    fn new(merges: Vec<(u32, u32)>, version: u32) -> PyResult<Self> {
        let pattern = pattern(version).ok_or_else(|| PyValueError::new_err("Unknown BPE pattern version"))?;
        let mut ranks = HashMap::with_capacity_and_hasher(merges.len(), Default::default());
        for (rank, &(a, b)) in merges.iter().enumerate() {
            let next = FIRST_MERGE + rank as u32;
            let special = |i: u32| (256..FIRST_MERGE).contains(&i);
            if next > u16::MAX as u32 || a >= next || b >= next || special(a) || special(b)
                || ranks.insert((a as u16, b as u16), (rank as u32, next as u16)).is_some() {
                return Err(PyValueError::new_err("Invalid BPE merge vocabulary"));
            }
        }
        Ok(Tokenizer { pattern: Regex::new(&pattern).expect("valid pattern"), ranks, cache: Mutex::new(HashMap::default()) })
    }

    #[getter]
    fn vocab_size(&self) -> u32 {
        FIRST_MERGE + self.ranks.len() as u32
    }

    fn encode(&self, text: &str) -> Vec<u16> {
        let mut out = Vec::new();
        self.extend(text, &mut out);
        out
    }


    fn encode_documents<'py>(&self, py: Python<'py>, texts: Vec<Option<String>>) -> Bound<'py, PyBytes> {
        let mut out = Vec::new();
        for text in texts.iter().flatten().filter(|text| !text.is_empty()) {
            out.push(BOS);
            self.extend(text, &mut out);
            out.push(EOS);
        }
        let bytes: Vec<u8> = out.iter().flat_map(|token| token.to_le_bytes()).collect();
        PyBytes::new(py, &bytes)
    }
}

#[pymodule]
fn kani_tokenize(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Tokenizer>()
}
