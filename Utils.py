from Configs.train_config import config
from tqdm import tqdm
import torch


# dataset

dataset_tokens = list(config['data_to_use'].keys()) + ['<comment>']


def convert_data_to_text(data_object, max_length=768, end_of_text_token="<|endoftext|>"):
    (FEN, moves, last_move_desc, legal_moves, attackers_list, attacks_list, comment) = data_object
    (FEN, moves, last_move_desc, legal_moves,
     attackers_list, attacks_list, comment) = (FEN[:max_length], moves[:max_length], last_move_desc[:max_length],
                                               legal_moves[:max_length], attackers_list[:max_length],
                                               attacks_list[:max_length], comment[:max_length])
    token_to_data = {'<fen>': FEN, '<moves>': moves, '<last move description>': last_move_desc,
                     '<legal moves>': legal_moves, '<attacked by>': attackers_list, '<attacks>': attacks_list}
    text = ""
    for token in token_to_data.keys():
        if config['data_to_use'][token]:
            text += f"{token} {token_to_data[token]} "
    text += f"<comment> {comment} {end_of_text_token}"  # comment always included at the end + end token

    return text


def get_chess_tokens():
    # squares
    board_notation = []
    col_names = 'abcdefgh'
    row_names = '87654321'
    for col in col_names:
        for row in row_names:
            board_notation.append(col+row)

    # moves
    moves = []
    # pieces = 'KQRBN'
    # for piece in pieces:
    #   for cell in board_notation:
    #     moves.append(piece+cell) # Ng4
    #     moves.append(piece+"x"+cell) # Qxf6
    #     for col in col_names:
    #       moves.append(piece+col+cell) # Nbc6
    #       moves.append(piece+col+"x"+cell) # Ngxe7
    #     for target_cell in board_notation:
    #       if cell != target_cell:
    #         moves.append(piece+cell+target_cell) # Ra5a6
    #         moves.append(piece+cell+"x"+target_cell) # Re6xe7

    # check + checkmate
    check_moves = []
    # for move in moves:
    #   check_moves.append(move+"+") # Nxg2+
    #   check_moves.append(move+"#") # Rd2#

    chess_vocab = board_notation + moves + check_moves + ["O-O"]

    return chess_vocab


# evaluation

def get_targets_and_outputs(model, dataset, comment_encoding, pad_token_id, max_length=768, eof='<|endoftext|>'):
    target_texts = []
    output_texts = []
    with tqdm(total=len(dataset)) as pbar:
        for idx, entry in enumerate(dataset):

          textual_data = model.tokenizer.decode(token_ids=entry[0], skip_special_tokens=False)
          textual_data = textual_data.split('<comment>')[1].split(eof)[0]
          target_texts.append(textual_data)

          comment_idx = list(entry[0]).index(comment_encoding) + 1
          input_encoding = entry[0][:comment_idx].unsqueeze(0).cuda()
          with torch.no_grad():
              outputs = model.model.generate(input_encoding, num_beams=2, no_repeat_ngram_size=2, max_length=max_length+1, pad_token_id=pad_token_id)
              output_text = model.tokenizer.decode(outputs[0], skip_special_tokens=False)
              output_text = output_text.split('<comment>')[1].split(eof)[0]
          output_texts.append(output_text)

          pbar.update(1)
    return target_texts, output_texts
