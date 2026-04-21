from transformers import DonutProcessor

proc = DonutProcessor.from_pretrained('Final_Trained_Model_files', local_files_only=True)
print('bos', proc.tokenizer.bos_token, proc.tokenizer.bos_token_id)
print('eos', proc.tokenizer.eos_token, proc.tokenizer.eos_token_id)
print('pad', proc.tokenizer.pad_token, proc.tokenizer.pad_token_id)
for token in ['<s_invoice>', '<s_iitcdip>', '<s_synthdog>', '<sep/>', '<s>']:
    ids = proc.tokenizer.encode(token, add_special_tokens=False)
    print(token, ids, proc.tokenizer.convert_ids_to_tokens(ids))
