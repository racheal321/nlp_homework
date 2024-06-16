import jieba
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import load_corpus
from model import Encoder, Decoder
from Attention import BahdanauAttention
import time
import os

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([vocab.token2id['<bos>']] * BATCH_SIZE, 1)

        # 教师强制 - 将目标词作为下一个输入
        for t in range(1, targ.shape[1]):
            # 将编码器输出 （enc_output） 传送至解码器
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)

            # 使用教师强制
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

def preprocess_sentence(sentence):
    sentence = jieba.lcut(sentence)
    return ['<bos>'] + sentence + ['<eos>'] + ['<pad>'] * (max_text_len - len(sentence))

def evaluate(sentence):
    attention_plot = np.zeros((max_text_len, max_text_len))

    sentence = preprocess_sentence(sentence)

    inputs = list()
    token_keys = vocab.keys()
    for token in sentence:
        if token in token_keys:
            inputs.append(vocab.token2id[token])
        else:
            inputs.append(vocab.token2id['<unk>'])
    # inputs = [vocab.token2id[i] for i in sentence]
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units)), tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([vocab.token2id['<bos>']], 0)

    invaild_id = [vocab.token2id['<pad>'], vocab.token2id['<bos>']]

    for t in range(max_text_len):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

        # 存储注意力权重以便后面制图
        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        top_3_index = tf.math.top_k(predictions[0], k=3).indices.numpy()
        for index in top_3_index:
            if index not in invaild_id:
                predicted_id = index
                break

        result += vocab[predicted_id] + ' '

        if vocab[predicted_id] == '<eos>':
            return result, sentence, attention_plot

        # 预测的 ID 被输送回模型
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot

# 注意力权重制图函数
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def translate(sentence):
    result, sentence, attention_plot = evaluate(sentence)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))

    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence)]
    plot_attention(attention_plot, sentence, result.split(' '))



if __name__ == '__main__':

    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val, vocab, max_text_len = load_corpus.get_dataset('jyxstxtqj_downcc/')

    vocab_size = vocab.__len__()
    BATCH_SIZE = 64
    embedding_dim = 16  #
    units = 128
    BUFFER_SIZE = len(input_tensor_train)
    steps_per_epoch = len(input_tensor_train) // BATCH_SIZE

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    example_input_batch, example_target_batch = next(iter(dataset))

    encoder = Encoder(vocab_size, embedding_dim, units, BATCH_SIZE)

    # 样本输入
    sample_hidden = encoder.initialize_hidden_state()
    sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
    print('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
    print('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden[0].shape))
    print('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden[1].shape))

    attention_layer = BahdanauAttention(10)
    attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

    print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
    print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

    decoder = Decoder(vocab_size, embedding_dim, units, BATCH_SIZE)
    sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)), sample_hidden, sample_output)

    print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

    optimizer = tf.keras.optimizers.Adam()

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

    EPOCHS = 10
    print("*************训练开始*************")
    for epoch in range(EPOCHS):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
        # 每 2 个周期（epoch），保存（检查点）一次模型
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    # 恢复检查点目录 （checkpoint_dir） 中最新的检查点
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    print("*************训练结束*************")

    translate(u'江湖恩怨，刀光剑影，一场江湖豪情。')  #chatGPT生成的金庸风格的文本