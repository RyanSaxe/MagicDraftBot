# MagicDraftBot

My initial attempt at a Magic: the Gathering (MTG) Draft AI. 

### Game Description

MTG Draft is a game where 8 players sit at a table and each open a pack of 14 MTG cards. They then select one (hidden), and pass the remainder (hidden) to the person to their left. This is repeated until no cards are left, and then repeated two more times (with two more packs) until each player has 42 cards. Then those 42 cards are used to construct a deck to play with. 

*Note: I currently don't have data corresponding to decks built nor games played, so this agent only participates in the card selection (drafting) process.*

### Algorithmic Structure

1. Cluster the dataset into archetypes via Kmeans
1. for each cluster, learn pick-order given data of P2P2 - P3P14
1. Initialize model weights with the normalized learned pick-orders
    1. Compute the bias towards each archetype given the cards in the pool
    1. Elevate the archetypal bias by adding a decaying function (simulated staying open)
1. Use this final archetypal bias to select the best card in the pack

I use L2 regularization on the weights in attempt to avoid exploding weights. However this doesn't appear to do enough to fight against an overly strong bias towards one color. I attempted to put a ceiling on the bias, however this led to the bot refusing to commit to an archetype and/or never pass rares. Still working on a solution to this. 

*Note: I cannot share the trained model nor any of the data used to train it.*

### Results (Accuracy on Human Picks)

The plot below displays the resulting accuracy on the test set. The blue line is the actual accuracy of the bot, where the yellow line is the accuracy if the bot's first or second choice was what the human selected.

![Accuracy Plot](https://raw.githubusercontent.com/RyanSaxe/MagicDraftBot/master/Output/Images/top2_accuracy_curve.png)

### External Content about this Agent:

* [Lords of Limited Podcast Episode (@16:25)](https://lordsoflimited.libsyn.com/lords-of-limited-129-bot-design-with-ryan-saxe)

### Next Step: Add More Features

Currently the bot does not look at things like converted mana-cost. CMC is inherrently encoded in the value of cards, however it is important for the bot to understand that even if a 5-cmc card is "better" than a 2-cmc card, it should take the cheaper card if it already has expensive cards.

Furthermore, the bot only knows synergy as it relates to archetypes, not cards. It takes Opt higher if it has Mad Ratter, and this is because it knows Opt and Mad Ratter are high in Izzet. But it doesn't know that Opt is specifically good with Mad Ratter. Creating a card-to-card synergy matrix could help the bot take non-blue cards that draw additional cards if there is a Mad Ratter in the pool, which is not something it can currently do.

### Example Drafts

Below are some decks and drafts that the first iteration of this bot did on MTGO. This was with a different modeling of staying open, and no bias ceiling, but a similar implementation. I will update this section as I continue to test the bot.

Draft: https://magic.flooey.org/draft/show?id=1_fn_hicUYtWQyvXSbjjhtsIy1k

![3-0](https://pbs.twimg.com/media/ELFSpb4XkAAYCrb?format=jpg&name=small)

Draft: https://magic.flooey.org/draft/show?id=9GymyrTy70YDYfHiKh2HRvNodao

![0-0](https://pbs.twimg.com/media/ELFUnzvWkAEFtGn?format=jpg&name=small)

Draft: https://magic.flooey.org/draft/show?id=xg0h8Jo41w2TcHNwwAQ-_UlxxFQ

![0-0](https://pbs.twimg.com/media/ELT3FbHW4AA3V9z?format=jpg&name=small)

### Full Table of Bots

Below are links to a draft where every seat at the table was my bot:

seat 1: https://magic.flooey.org/draft/show?id=Y66phkMHXy1Fkxinlv2_0mgnyFM

seat 2: https://magic.flooey.org/draft/show?id=ISj69cO7itncFby65oNGRxzSsSM

seat 3: https://magic.flooey.org/draft/show?id=7W_eV52n2LhQay9tywaEUaypoGw

seat 4: https://magic.flooey.org/draft/show?id=8S0w_5hMokFeH8rCvxcER_BKZJU

seat 5: https://magic.flooey.org/draft/show?id=PPy7SadVNNPVCuELgdkYNLWrocI

seat 6: https://magic.flooey.org/draft/show?id=b6Rcmynu0iEWBtsrC90gQgrtEV8

seat 7: https://magic.flooey.org/draft/show?id=Tyhg2f-8J08rOgaU6TaQdkq4i7Y

seat 8: https://magic.flooey.org/draft/show?id=jfjbVeeFaHzTuVde2za0axX2W20

### Future Goals

Right now this is modeled such that there is no integration with the whole table; no reading of signals. Eventually I would like to design a Long Short Term Memory (LSTM) Network that is optimized for predicting what the people next to the bot are drafting. This could then be integrated into the current model to help bias away from archetypes that appear closed and towards archetypes that appear open!

Overall, a greedy-optimization system is not the best way to solve this problem. However, it's hard to do much more with the data I have. Eventually I would like to explore reinforcement learning in this space, but that will also require a deck evaluation function. It's far in the future, but something I will do when I get everything in place.

### File Descriptions

models.py --- the two models trained to create the Draft Agent

preprocessing.py --- some data processing techniques (e.g. label drafts via Kmeans)

create_model.py --- the script for creating the model

predict_from_log.py --- script for seeing how the bot would navigate a draft given a MTGO Draft Log

bot_table_draft.py --- script for generating a draft with 8 bots at the table

Output/ place to store images and files to show results. 

Data/ and Saved_Models/ --- folders in .gitignore as these cannot be in this repo due to NDA.
