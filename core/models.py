"""
Module: core.models
Deskripsi: Menyediakan definisi arsitektur jaringan saraf tiruan deep learning
           (Keras Deep Neural Networks) untuk klasifikasi beban kognitif subjek.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Bidirectional, Dense, Dropout, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay

def build_cnn_lstm_tabular(input_dim=3, learning_rate=0.001):
    """
    Mengonstruksi arsitektur hybrid jaringan saraf konvolusi (Conv1D) dan 
    Long Short-Term Memory (LSTM) untuk data input tabular berupa fitur statistik.

    Parameters:
    -----------
    input_dim : int, default=3
        Jumlah dimensi fitur input (jumlah baris fitur dalam urutan sekuensial).
    learning_rate : float, default=0.001
        Laju pembelajaran inisial untuk optimizer Adam.

    Returns:
    --------
    tensorflow.keras.Sequential
        Objek model Keras terkompilasi.
    """
    model = Sequential()
    # Pengaturan kernel_size=2 disesuaikan untuk ekstraksi fitur spasial pada matriks data berdimensi rendah
    model.add(Conv1D(filters=32, kernel_size=2, activation="relu", input_shape=(input_dim, 1)))
    model.add(LSTM(64, activation="tanh"))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_lstm_attention_network(input_dim=3, learning_rate=0.001):
    """
    Mengonstruksi model kombinasi lapisan rekuren LSTM dengan mekanisme Multi-Head Attention
    untuk menangkap dependensi temporal jangka panjang yang bersifat dinamis.

    Parameters:
    -----------
    input_dim : int, default=3
        Jumlah dimensi fitur input.
    learning_rate : float, default=0.001
        Laju pembelajaran inisial untuk optimizer Adam.

    Returns:
    --------
    tensorflow.keras.Model
        Objek model fungsional Keras terkompilasi.
    """
    inputs = Input(shape=(input_dim, 1))
    
    # Flag return_sequences=True wajib diaktifkan agar struktur dimensi temporal dipertahankan untuk Attention Layer
    lstm_out = LSTM(32, return_sequences=True, activation="tanh")(inputs)
    
    # Implementasi Multi-Head Attention Layer
    attn_out = MultiHeadAttention(num_heads=2, key_dim=32)(query=lstm_out, value=lstm_out)
    
    # Pengurangan dimensi menggunakan Global Average Pooling sebelum klasifikasi linear dense
    pool_out = GlobalAveragePooling1D()(attn_out)
    drop_out = Dropout(0.2)(pool_out)
    dense_out = Dense(16, activation="relu")(drop_out)
    outputs = Dense(1, activation="sigmoid")(dense_out)
    
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_cnn_bilstm_waveform(window_size=1280, learning_rate=0.001, use_cosine_decay=False, total_steps=10000):
    """
    Mengonstruksi arsitektur Waveform Processor tingkat tinggi berbasis Deep Conv1D
    dan Bidirectional LSTM (BiLSTM) dua arah untuk ekstraksi pola langsung dari sinyal gelombang mentah.

    Parameters:
    -----------
    window_size : int, default=1280
        Panjang urutan sekuensial atau jumlah titik sampling gelombang dalam satu window.
    learning_rate : float, default=0.001
        Laju pembelajaran inisial.
    use_cosine_decay : bool, default=False
        Flag aktivasi penjadwalan laju pembelajaran berbasis fungsi kosinus (Cosine Learning Rate Decay).
    total_steps : int, default=10000
        Total langkah pembusukan laju belajar (decay steps).

    Returns:
    --------
    tensorflow.keras.Sequential
        Objek model sekuensial Keras terkompilasi.
    """
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=5, activation="relu", input_shape=(window_size, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    
    # Penambahan lapisan Bidirectional LSTM untuk membaca pola arah maju (forward) dan mundur (backward)
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.2))
    
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    
    if use_cosine_decay:
        lr_schedule = CosineDecay(initial_learning_rate=learning_rate, decay_steps=total_steps)
        optimizer = Adam(learning_rate=lr_schedule)
    else:
        optimizer = Adam(learning_rate=learning_rate)
        
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_multimodal_fusion_network(input_dim=6, learning_rate=0.001):
    """
    Mengonstruksi arsitektur hybrid model fusi multimodal tingkat fitur (Feature-Level Multimodal Fusion)
    untuk mengolah kombinasi data multi-sensor (ECG dan PPG secara simultan).

    Parameters:
    -----------
    input_dim : int, default=6
        Jumlah total dimensi fitur input terintegrasi (misal: 3 fitur ECG + 3 fitur PPG).
    learning_rate : float, default=0.001
        Laju pembelajaran untuk optimizer Adam.

    Returns:
    --------
    tensorflow.keras.Sequential
        Objek model sekuensial fusi Keras terkompilasi.
    """
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=2, activation="relu", input_shape=(input_dim, 1)))
    model.add(LSTM(64, activation="tanh"))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model