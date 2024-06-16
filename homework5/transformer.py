import os
import time
import tensorflow as tf
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
import re

# start_time = time.time()
#
# # data_dir = 'D:/zongruntang/nlp/second/jyxstxtqj_downcc.com/'
# # files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
#
# texts = []
# # for file in files:
# #     with open(file, encoding='ansi', errors='ignore') as f:
# #         texts.append(f.read())
# book_list = ['笑傲江湖','神雕侠侣']
# content = ''
# for book_name in book_list:
#     f = open("D:/zongruntang/nlp/second/jyxstxtqj_downcc.com/" + book_name + ".txt", "r", encoding='gbk', errors='ignore')
#     texts = f.read()
#
# dataset = Dataset.from_dict({"text": texts})
#
# class CustomTokenizer:
#     def __init__(self, texts):
#         self.vocab = self.build_vocab(texts)
#         self.vocab_size = len(self.vocab)
#         self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
#         self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
#
#     def build_vocab(self, texts):
#         tokens = set()
#         for text in texts:
#             tokens.update(re.findall(r'\w+|\S', text))
#         return sorted(tokens)
#
#     def encode(self, text):
#         return [self.token_to_id[token] for token in re.findall(r'\w+|\S', text)]
#
#     def decode(self, token_ids):
#         return ''.join([self.id_to_token[token_id] for token_id in token_ids])
#
# tokenizer = CustomTokenizer(texts)
#
# # def tokenize_function(examples):
# #     return {"input_ids": [tokenizer.encode(text) for text in examples["text"]]}
# def tokenize_function(examples):
#     text = examples["text"][0]
#     token_ids = tokenizer.encode(text)
#     # 将长序列切分成长度为512的块
#     sequences = [token_ids[i:i+512] for i in range(0, len(token_ids), 512)]
#     return {"input_ids": sequences}
# # tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"], batch_size=1)
# tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
# input_ids = [sequence for sequences in tokenized_datasets['input_ids'] for sequence in sequences]
# max_length = 512
# input_ids = tf.keras.preprocessing.sequence.pad_sequences(
#     [x for x in tokenized_datasets['input_ids']], maxlen=max_length, padding='post'
# )
#
# train_input_ids, val_input_ids = train_test_split(input_ids, test_size=0.1)
#
# train_dataset = tf.data.Dataset.from_tensor_slices((train_input_ids, train_input_ids)).shuffle(len(train_input_ids)).batch(4)
# val_dataset = tf.data.Dataset.from_tensor_slices((val_input_ids, val_input_ids)).batch(4)
#
# class TransformerModel(tf.keras.Model):
#     def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, max_seq_length=512):
#         super(TransformerModel, self).__init__()
#         self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
#         self.positional_encoding = self.add_weight(name="positional_encoding", shape=(1, max_seq_length, d_model), initializer='zeros')
#         self.encoder_layers = [
#             tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model//nhead)
#             for _ in range(num_layers)
#         ]
#         self.decoder_layers = [
#             tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model//nhead)
#             for _ in range(num_layers)
#         ]
#         self.fc_out = tf.keras.layers.Dense(vocab_size)
#
#     def call(self, inputs, training=False, tgt=None):
#         print(inputs)
#         print("inputs[0].shape: ", inputs[0].shape)
#         src = self.embedding(inputs[0]) + self.positional_encoding[:, :inputs[0].shape[1], :]
#         src = self.embedding(inputs[0]) + self.positional_encoding[:, :inputs[0].shape[1], :]
#         memory = src
#         for encoder_layer in self.encoder_layers:
#             memory = encoder_layer(memory, memory, return_attention_scores=False, training=training)
#         output = tgt
#         if output is None:
#             return None
#         for decoder_layer in self.decoder_layers:
#             output = decoder_layer(output, memory, return_attention_scores=False, training=training)
#         return self.fc_out(output)
#
#
#
# vocab_size = tokenizer.vocab_size
# model = TransformerModel(vocab_size)
#
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
# device = '/gpu:0'
# with tf.device(device):
#     model.fit(train_dataset, validation_data=val_dataset, epochs=3)
#
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Training completed in: {elapsed_time // 3600}h {(elapsed_time % 3600) // 60}m {elapsed_time % 60}s")
#
# model.save_weights("./transformer-finetuned-novels-tf.h5")
# np.save("./transformer-finetuned-novels-tf-vocab.npy", tokenizer.vocab)
#
# # 生成文本
# def generate_text(model, tokenizer, prompt, max_length=200):
#     input_ids = tokenizer.encode(prompt)
#     output_ids = input_ids
#     for _ in range(max_length):
#         output = model(tf.constant([output_ids]), tf.constant([output_ids]))
#         next_token_id = tf.argmax(output[0, -1, :])
#         output_ids.append(next_token_id.numpy())
#         if next_token_id == tokenizer.token_to_id.get('[SEP]', -1):
#             break
#     return tokenizer.decode(output_ids)
#
# # 输入小说片段
# prompt = "令狐冲淡然一笑，道：令狐冲死在姑娘的言语之下，那也不错啊。"
#
# generated_text = generate_text(model, tokenizer, prompt)
# print(generated_text)
import os
import time
import tensorflow as tf
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
import re

start_time = time.time()

# 替换为你的实际数据目录
data_dir = 'D:/zongruntang/nlp/second/jyxstxtqj_downcc.com/'

# 要读取的书籍列表
book_list = ['笑傲江湖']

# 从文件中读取文本
texts = []
for book_name in book_list:
    file_path = os.path.join(data_dir, f"{book_name}.txt")
    with open(file_path, "r", encoding='gbk', errors='ignore') as f:
        texts.append(f.read())  # 将每个文本追加到列表中

# 将文本数据转换为数据集
dataset = Dataset.from_dict({"text": texts})

class CustomTokenizer:
    def __init__(self, texts):
        self.vocab = self.build_vocab(texts)
        self.vocab_size = len(self.vocab)
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def build_vocab(self, texts):
        tokens = set()
        for text in texts:
            tokens.update(re.findall(r'\w+|\S', text))  # 使用正则表达式找到所有单词和符号
        return sorted(tokens)

    def encode(self, text):
        return [self.token_to_id[token] for token in re.findall(r'\w+|\S', text)]  # 编码文本

    def decode(self, token_ids):
        return ''.join([self.id_to_token[token_id] for token_id in token_ids])  # 解码文本

tokenizer = CustomTokenizer(texts)

# 分词函数，将文本分割为长度为512的块
def tokenize_function(examples):
    text = examples["text"]
    token_ids = tokenizer.encode(text)
    sequences = [token_ids[i:i+512] for i in range(0, len(token_ids), 512)]
    return {"input_ids": sequences}

# 修改这里，将单个文本拆分为多个示例
def split_text_into_examples(texts, max_length=512):
    examples = []
    for text in texts:
        token_ids = tokenizer.encode(text)
        sequences = [token_ids[i:i+max_length] for i in range(0, len(token_ids), max_length) if len(token_ids[i:i+max_length]) > 0]
        examples.extend(sequences)
    return examples

# 拆分文本数据
input_ids = split_text_into_examples(texts)

max_length = 512
# 使用pad_sequences函数将所有序列填充到相同长度
input_ids = tf.keras.preprocessing.sequence.pad_sequences(input_ids, maxlen=max_length, padding='post')

# 将数据集分为训练集和验证集
train_input_ids, val_input_ids = train_test_split(input_ids, test_size=0.1)

# 检查数据集是否有None值或空值
assert all(len(seq) == max_length for seq in train_input_ids), "训练集包含不一致长度的序列"
assert all(len(seq) == max_length for seq in val_input_ids), "验证集包含不一致长度的序列"

# 创建TensorFlow数据集
train_dataset = tf.data.Dataset.from_tensor_slices((train_input_ids, train_input_ids)).shuffle(len(train_input_ids)).batch(4)
val_dataset = tf.data.Dataset.from_tensor_slices((val_input_ids, val_input_ids)).batch(4)

# 打印检查数据集中的一些样本
for input_id, target_id in train_dataset.take(1):
    print("Input IDs:", input_id.numpy())
    print("Target IDs:", target_id.numpy())

class TransformerModel(tf.keras.Model):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, max_seq_length=512):
        super(TransformerModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.positional_encoding = self.positional_encoding_layer(max_seq_length, d_model)  # 位置编码
        self.encoder_layers = [
            tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model//nhead)
            for _ in range(num_layers)
        ]
        self.decoder_layers = [
            tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model//nhead)
            for _ in range(num_layers)
        ]
        self.fc_out = tf.keras.layers.Dense(vocab_size)

    # 位置编码函数
    def positional_encoding_layer(self, max_seq_length, d_model):
        pos = np.arange(max_seq_length)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        angle_rads = pos * angle_rates

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])  # 偶数位置的正弦
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])  # 奇数位置的余弦

        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs, training=False, tgt=None):
        src = self.embedding(inputs) + self.positional_encoding[:, :inputs.shape[1], :]  # 添加位置编码
        memory = src
        for encoder_layer in self.encoder_layers:
            memory = encoder_layer(query=memory, value=memory, key=memory, return_attention_scores=False, training=training)
        if tgt is None:
            tgt = inputs
        output = self.embedding(tgt) + self.positional_encoding[:, :tgt.shape[1], :]
        for decoder_layer in self.decoder_layers:
            output = decoder_layer(query=output, value=memory, key=memory, return_attention_scores=False,
                                   training=training)
        for decoder_layer in self.decoder_layers:
            output = decoder_layer(query=output, value=memory, key=memory, return_attention_scores=False, training=training)
        return self.fc_out(output)

vocab_size = tokenizer.vocab_size
model = TransformerModel(vocab_size)

# 确保模型的所有层都构建
model.build(input_shape=(None, max_length))

# 编译和训练模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

device = '/gpu:0'
with tf.device(device):
    model.fit(train_dataset, validation_data=val_dataset, epochs=3)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training completed in: {elapsed_time // 3600}h {(elapsed_time % 3600) // 60}m {elapsed_time % 60}s")

# 保存模型权重和词汇表
model.save_weights("./transformer-finetuned-novels-tf.weights.h5")
np.save("./transformer-finetuned-novels-tf-vocab.npy", tokenizer.vocab)

# 生成文本函数
def generate_text(model, tokenizer, prompt, max_length=200, temperature=1.0, top_k=50, repetition_penalty=1.2):
    input_ids = tokenizer.encode(prompt)
    output_ids = input_ids.copy()

    for _ in range(max_length):
        output = model(inputs=tf.constant([output_ids]), tgt=tf.constant([output_ids]), training=False)

        # Retrieve logits from the last position
        logits = output[0, -1, :]

        # Apply temperature to logits
        logits /= temperature

        # Apply top-k sampling
        logits, top_k_indices = tf.math.top_k(logits, k=top_k)
        top_k_probs = tf.nn.softmax(logits)

        # Convert logits to probabilities
        probs = tf.nn.softmax(logits).numpy()

        # Apply repetition penalty
        for token_id in set(output_ids):
            if token_id in top_k_indices:
                probs[token_id] /= repetition_penalty

        # Normalize probabilities
        top_k_probs = probs[top_k_indices.numpy()]
        top_k_probs /= np.sum(top_k_probs)

        # Sample the next token from top-k probabilities
        next_token = np.random.choice(top_k_indices.numpy(), p=top_k_probs)

        output_ids.append(next_token)

        # Stop generation if [SEP] token is generated
        if next_token == tokenizer.token_to_id.get('[SEP]', -1):
            break

        # Check for repetitive generation in the last 10 tokens
        if len(output_ids) > 10 and len(set(output_ids[-10:])) < 4:
            output_ids = output_ids[:-1]  # Remove the last token to avoid repetition
            continue

    return tokenizer.decode(output_ids)


# 输入小说片段
prompt = "令狐冲淡然一笑，道：令狐冲死在姑娘的言语之下，那也不错啊。"

generated_text = generate_text(model, tokenizer, prompt)
print(generated_text)




