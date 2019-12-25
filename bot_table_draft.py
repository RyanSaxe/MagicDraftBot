from models import DraftNet
from utils import generate_pack
import pandas as pd
import random
import torch
import sys
import datetime
import os
from preprocessing import get_format_features
save_logs = sys.argv[1]
#load in model and name map
model = torch.load('Saved_Models/draft_model.pkl')
card_df = get_format_features()
#define draft with 8 players, 3 rounds, and packs of 14 cards
seats = 8
n_rounds = 3
n_sub_rounds = 14
n_cards = len(card_df)
#index circular shuffle per iteration
pack_shuffle_right = [7,0,1,2,3,4,5,6]
pack_shuffle_left = [1,2,3,4,5,6,7,0]
#initialize
for_draft_logs = torch.zeros(size=(seats,n_rounds * n_sub_rounds,n_cards * 2))
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
		bot_pick_names = [card_df.loc[bp.item()]['orig_name'] for bp in bot_picks]
		#store pick
		names.append(bot_pick_names)
		#update bot pools
		for idx,bot_pick in enumerate(bot_picks):
			bp = bot_pick.item()
			pick_encoded = torch.zeros(n_cards)
			pick_encoded[bp] = 1
			for_draft_logs[idx,pick_n] = torch.cat([packs[idx],pick_encoded])
			packs[idx][bp] = 0
			picks[idx][bp] += 1
		#pass the packs (left, right, left)
		if larger_round % 2 == 1:
			packs = [packs[idx] for idx in pack_shuffle_right]
		else:
			packs = [packs[idx] for idx in pack_shuffle_left]
#display the draft picks
print(pd.DataFrame(names))

"""create_logs:"""
if save_logs.lower() == 'save':
	unique = str(datetime.datetime.now())
	logs = ["" for x in range(seats)]
	bot_names = ['bot' + str(i) for i in range(seats)]
	event_header = 'Event #: 1\nTime: ' + str(unique) + '\nPlayers:\n'
	for idx,bot in enumerate(for_draft_logs):
		pack_counter = 1
		bot_name_head = ['    ' + name if i != idx else '--> ' + name for i,name in enumerate(bot_names)]
		header = event_header + '\n'.join(bot_name_head)
		logs[idx] += header
		for pick_n,data in enumerate(bot):
			if pick_n % 14 == 0:
				pack_header = '\n\n------ Pack ' + str(pack_counter) + ': Throne of Eldraine ------'
				logs[idx] += pack_header
				pack_counter += 1
			m = (pick_n % n_sub_rounds) + 1
			n = (int(pick_n/n_sub_rounds)) + 1
			logs[idx] += "\n\n"
			logs[idx] += f"Pack {n} pick {m}:\n"
			pick,pack = data[n_cards:],data[:n_cards]
			names = ["--> " + card_df.loc[i]['orig_name'] if i == pick.argmax() else "    " + card_df.loc[i]['orig_name'] for i,c in enumerate(pack) if c == 1]
			logs[idx] += "\n".join(names)
	log_loc = 'Output/Logs/Generated_on_' + unique
	os.mkdir(log_loc)
	for i,log in enumerate(logs): 
		fname = log_loc + '/bot' + str(i) + '_' + unique + '.txt'
		with open(fname,'w') as f:
			f.write(log)