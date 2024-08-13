function summary = generate_summary_py(tokenizer, model, text)
    inputs = tokenizer(text, 'return_tensors', 'pt');
    input_ids = inputs.input_ids;
    attention_mask = inputs.attention_mask;

    summary_ids = model.generate(input_ids, 'max_length', 100, 'attention_mask', attention_mask);
    summary = tokenizer.decode(summary_ids, 'skip_special_tokens', true);
end