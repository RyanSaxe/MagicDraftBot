import torch
class RankingNet(torch.nn.Module):
    """
    Learn N pick orders (1 per archetypal cluster) for draft picks
    
    assumes the first item in the data is an archetype label.
    """
    def __init__(self,num_cards,num_archetypes):
        super(RankingNet, self).__init__()
        self.n_cards = num_cards
        self.n_archs = num_archetypes
        #goal is transparency. self.ratings.weights is the
        #   parameter we're trying to learn
        self.ranking_matrix = torch.nn.Parameter(
                                torch.rand(
                                    (self.n_cards,self.n_archs),
                                    requires_grad=True, 
                                    dtype=torch.float
                                ))
    def forward(self,x):
        #get the archetype label
        arch_idx = x[:,0].type(torch.long)
        #get the current options in the pack
        pack = x[:,1 + self.n_cards:]
        return self.ranking_matrix[:,arch_idx].t() * pack

class DraftNet(torch.nn.Module):
    """
    NN that learns how to navigate a draft with a given set of archetype initializations
    """
    def __init__(self,rank_matrix):
        """
        rank_matrix: pre-initialized m x n matrix where m is number of cards
                        in the set and n is the number of archetypes

        """
        super(DraftNet, self).__init__()
        self.n_cards,self.n_archs = rank_matrix.shape
        #m x n matrix where m is number of cards in the set 
        #and n is the number of archetypes. Conceptually, this
        #matrix helps dictate how to make decisions in order to
        #properly navigate a draft towards each archetype
        self.rank_matrix = torch.nn.Parameter(
                                        torch.tensor(
                                            rank_matrix,
                                            requires_grad=True,
                                            dtype=torch.float
                                        )
                                    )
        #vector to express opposition to bias (staying open)
        #initially this had a stronger decaying structure on top
        #but I wanted to see what happened when left alone.
        # -> structure still decays, but spikes for P2P1 and P3P1
        #    likely due to humans rare-drafting.
        #original decay structure was implemented in one of two ways:
        #   1. log(-mx + b)/log(-m + b) where x = pick_number/n
        #   2. flip(cumsum(self.open_base)) - self.open_base.min()
        self.open_base = torch.nn.Parameter(torch.tensor(
                                            torch.ones(42),
                                            requires_grad=True,
                                            dtype=torch.float
                                        ))
        #ceiling for the bias to avoid inflexibility later in the draft
        self.max_pull = torch.nn.Parameter(torch.tensor(
                                            5,
                                            requires_grad=True,
                                            dtype=torch.float
                                        ))
        #placeholder for future versions. Currently doesn't update, but I want to explore that.
        self.arch_bias = torch.nn.Parameter(torch.ones(self.n_archs,requires_grad=False))
        self.relu = torch.nn.ReLU()
    def forward(self,x):
        #gets pool of cards drafted so far
        pool = x[:,:self.n_cards]
        #gets current pack to decide what to pick from
        pack = x[:,self.n_cards:]
        #computes where in the draft it is. This is to index
        #into self.open_base. Clamp above 42 to enable using
        #the model as an oracle ('what if it had all the red cards').
        pick_n = torch.clamp(pool.sum(1),max=41)
        open_factor = self.relu(self.open_base[pick_n.type(torch.long)])
        #compute bias towards card drafted so far
        simple_pull = torch.matmul(pool,self.rank_matrix)
        pull_relu = self.relu(simple_pull)
        #I want to explore a version of this model where the pool informs
        #the pull as a probability distribution across archetypes, and then
        #has some memory of past cards seen to also update that distribution
        #based on what it models as open
        pull_w_open = (pull_relu * self.arch_bias) + open_factor[:,None]
        pull_thresh = self.relu(pull_w_open)
        #place the ceiling on the bias
        final_pull = torch.where(pull_thresh > self.max_pull, self.max_pull, pull_thresh)
        #rank every card in the format according to bias
        pick_rankings = torch.matmul(final_pull,self.rank_matrix.t())
        pick_relu = self.relu(pick_rankings)
        #zero value for all cards that are not in the current pack
        return pick_relu * pack