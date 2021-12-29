A Pytorch Implementation of the Transformer Network
---
**Transformer**

Pytorch implementations of "Attention is All You Need" (Vaswani et al., NIPS 2017) and "Weighted Transformer Network for Machine Translation" (Ahmed et al., arXiv 2017)

```python
# Example
model = make_model(src_vocab = 10, tgt_vocab=10, N=2)

src= torch.randint(1,10,(1,10))
trg= torch.randint(1,10,(1,9))
src_mask=torch.tensor([[True, True, True, True, True, True, True, True, True, True]])
trg_mask=torch.tensor(
    [[[ True, False, False, False, False, False, False, False, False],
      [ True,  True, False, False, False, False, False, False, False],
      [ True,  True,  True, False, False, False, False, False, False],
      [ True,  True,  True,  True, False, False, False, False, False],
      [ True,  True,  True,  True,  True, False, False, False, False],
      [ True,  True,  True,  True,  True,  True, False, False, False],
      [ True,  True,  True,  True,  True,  True,  True, False, False],
      [ True,  True,  True,  True,  True,  True,  True,  True, False],
      [ True,  True,  True,  True,  True,  True,  True,  True,  True]]])

print(src.shape) # [batch, src_len]
print(trg.shape) # [batch, trg_len]
print(src_mask.shape) # [batch, src_len]
print(trg_mask.shape) # [batch, trg_len, trg_len]

model(src,trg,src_mask, trg_mask)
```
<br/>

**BERT**

Pytorch implementations of "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (Devlin, Jacob, et al., arXiv 2018)


```python
# Example
input_ids = torch.LongTensor([[31, 51, 12, 23, 99], [15, 5, 1, 0, 0]])
attention_mask = torch.LongTensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
token_type_ids = torch.LongTensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])

net = BertModel(config)

encoded_layers, pooled_output, attention_probs = net(
    input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, attention_show_flg=True)

print("encoded_layers size :", encoded_layers.shape) # [batch, seq_len, hidden]
print("pooled_output size :", pooled_output.shape) # [batch, hidden]
print("attention_probs size :", attention_probs.shape) # [batch, num_layers, seq_len, seq_len]
```
<br/>

**GPT2**

Pytorch implementations of "Language Models are Unsupervised Multitask Learners" (Radford et al., arXiv 2018)

```python
# Example
src=torch.randint(10,(16,256))

GPT2=Transformer(layers=6, pad_idx=0, words=10000, seq_len=512, heads=6, dims=786)

logits=GPT2(src) # [batch, seq_len, vocab_size]
output_tokens=torch.argmax(nn.Softmax(-1)(logits),-1)

print(output_tokens.shape) # [batch, seq_len]
```

Reference
---
**Paper**

- Vaswani et al., "Attention is All You Need", NIPS 2017
- Ahmed et al., "Weighted Transformer Network for Machine Translation", Arxiv 2017
- Devlin, Jacob, et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
- Radford et al., "Language Models are Unsupervised Multitask Learners", ArXiv 2018)


**Code**

- Annotated Transformer - https://nlp.seas.harvard.edu/2018/04/03/attention.html
- Huggingface github - https://github.com/huggingface/transformers
- Huggingface BERT - https://huggingface.co/docs/transformers/model_doc/bert
- OPEN AI/GPT2 - https://github.com/openai/gpt-2
