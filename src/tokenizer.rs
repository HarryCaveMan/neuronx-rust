use tokenizers::{
    utils::{
        padding::{PaddingParams,PaddingStrategy},
        truncation::{TruncationParams,TruncationStrategy}
    },
    models::wordpiece::WordPiece,
    tokenizer::{Encoding,Result},
    Tokenizer
};
use ndarray::{
    aview1,
    Array2,
    Axis,
    parallel::{
        par_azip
    }
};
use std::time::Instant;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq)]
pub struct BatchSentenceEncoding {
    pub input_ids: Array2<u32>,
    pub token_type_ids: Array2<u32>,
    pub attention_mask: Array2<u32>
}

unsafe impl Send for  BatchSentenceEncoding {}
unsafe impl Sync for BatchSentenceEncoding {}

impl BatchSentenceEncoding {
    pub fn empty(nrows: usize,ncols: usize) -> Self {
        Self {
            input_ids: Array2::<u32>::zeros((nrows,ncols)),
            token_type_ids: Array2::<u32>::zeros((nrows,ncols)),
            attention_mask: Array2::<u32>::zeros((nrows,ncols))
        }
    }

    pub fn from_hf_batch_par(batch: Vec<Encoding>) -> Result<Self> {
        let nrows = batch.len();
        let ncols = batch[0].get_ids().len();
        let mut batch_encodings = Self::empty(nrows,ncols);
        par_azip!(
            (
                mut input_chunk in batch_encodings.input_ids.rows_mut(),
                mut type_chunk in batch_encodings.token_type_ids.rows_mut(),
                mut mask_chunk in batch_encodings.attention_mask.rows_mut(),
                encoding in &batch
            )
            {
                input_chunk.assign(&aview1(encoding.get_ids()));
                type_chunk.assign(&aview1(encoding.get_type_ids()));
                mask_chunk.assign(&aview1(encoding.get_attention_mask()));
            }
        );
        Ok(batch_encodings)
    }

    pub fn from_hf_batch_serial(batch: Vec<Encoding>) -> Result<Self> {
        let nrows = batch.len();
        let ncols = batch[0].get_ids().len();
        let mut batch_encodings = Self::empty(nrows,ncols);
        batch.iter().enumerate().for_each(|(i,encoding)| {
            batch_encodings.input_ids.row_mut(i).assign(&aview1(encoding.get_ids()));
            batch_encodings.token_type_ids.row_mut(i).assign(&aview1(encoding.get_type_ids()));
            batch_encodings.attention_mask.row_mut(i).assign(&aview1(encoding.get_attention_mask()));
        });
        Ok(batch_encodings)
    }

    pub fn from_hf_batch(batch: Vec<Encoding>, par_thresh: usize) -> Result<Self> {
        let nrows = batch.len();
        if nrows >= par_thresh {
            Self::from_hf_batch_par(batch)
        } else {
            Self::from_hf_batch_serial(batch)
        }
    }
}

pub struct BatchSentenceTokenizer {
    tokenizer: Tokenizer
}

impl BatchSentenceTokenizer {
    pub fn new(config_file: PathBuf, truncation_strategy: TruncationStrategy,padding_strategy: PaddingStrategy) -> Result<Self> {
        let mut padding_params = PaddingParams::default();
        padding_params.strategy =  padding_strategy;
        let mut truncation_params = TruncationParams::default();
        truncation_params.strategy = truncation_strategy;
        let mut tokenizer = Tokenizer::from_file(config_file)?;
        tokenizer.with_padding(Some(padding_params)).with_truncation(Some(truncation_params))?;
        Ok(Self { tokenizer })
    }

    pub fn batch_encode_sentences(&self, sentences: &[&str]) -> Result<Vec<BatchSentenceEncoding>> {
        let encoded_batch = self.tokenizer.encode_batch(sentences, false)?;
        Ok(BatchSentenceEncoding::from_hf_batch_par(encoded_batch)?)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;

    #[test]
    fn test_batch_sentence_encoding() -> Result<()> {
        let batch = vec![
            "He pioneered advances in renewable energy technology.",
            "His inventions made solar power more accessible.",
            "Awards and recognition followed his groundbreaking work.",
            "His legacy inspired future generations of scientists.",
            "They organized a festival celebrating cultural diversity.",
            "Music, dance, and cuisine from around the world were featured.",
            "The event fostered understanding and appreciation among attendees.",
            "It became an annual tradition cherished by the community.",
            "Technological integration in healthcare improved patient outcomes.",
        ];
        let tokenizer = BatchSentenceTokenizer::new(PathBuf::from("path/to/config.json"), TruncationStrategy::LongestFirst, PaddingStrategy::Longest)?;
        let batch_encodings = tokenizer.batch_encode_sentences(&batch)?;
        assert_eq!(batch_encodings.input_ids.shape(), (3, 10)); // Example shape
        Ok(())
    }
}