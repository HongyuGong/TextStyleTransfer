#extra_words = ["<TOS>", "<EOS>", "<UNK>"]
extra_words = ["<TOS>", "<EOS>", "<UNK>"]
start_token = extra_words.index("<TOS>")
end_token = extra_words.index("<EOS>")
unk_token = extra_words.index("<UNK>")


# weights
style_weight = 1.0
semantic_weight = 0.4 #0.4
lm_weight = 0.1 #0.8
