config = {
    'Model': 'gpt2',
    'NUMER_OF_DATA_DIRS': 12,
    'batch_size': 2,
    'lr':  3e-5,
    'train_precentege': 0.9,
    'epochs': 100,
    'data_to_use': {'<fen>': True, '<moves>': True, '<last move description>': True,
                    '<legal moves>': False, '<attacked by>': True, '<attacks>': True}
}
