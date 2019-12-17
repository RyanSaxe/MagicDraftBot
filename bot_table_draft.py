from models import DraftNet
from utils import generate_pack
import pandas as pd
import random
import torch
#load in model and name map
model = torch.load('Saved_Models/draft_model.pkl')
card_df = torch.load('Data/ft.pkl')
#define draft with 8 players, 3 rounds, and packs of 14 cards
seats = 8
n_rounds = 3
n_sub_rounds = 14
n_cards = len(card_df)
#index circular shuffle per iteration
pack_shuffle = [7,0,1,2,3,4,5,6]
#initialize
names = []
picks = [torch.zeros(n_cards) for pack in range(seats)]
for larger_round in range(n_rounds):
	#generate packs for this round
	packs = [generate_pack(card_df) for pack in range(seats)]
	for smaller_round in range(n_sub_rounds):
		pick_n = n_sub_rounds * larger_round + smaller_round
		#get data for each bot
		data = torch.stack([torch.cat([picks[idx],packs[idx]]) for idx in range(seats)])
		#make pick
		bot_picks = model(data).argmax(1)
		bot_pick_names = [card_df.loc[bp.item()]['name'] for bp in bot_picks]
		#store pick
		names.append(bot_pick_names)
		#update bot pools
		for idx,bot_pick in enumerate(bot_picks):
			bp = bot_pick.item()
			pick_encoded = torch.zeros(n_cards)
			pick_encoded[bp] = 1
			packs[idx][bp] = 0
			picks[idx][bp] += 1
		#pass the packs
		packs = [packs[idx] for idx in pack_shuffle]
#display the draft picks
print(pd.DataFrame(names))