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
        self.rank_matrix = torch.nn.Parameter(
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
        return self.rank_matrix[:,arch_idx].t() * pack

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
        #the initialization of 2 as open_base and 7 is purposeful.
        #This is because cumprod(sigmoid(open_base)) * 7 - minimum has a value
        # of tensor(1.0090) at Pack 2 Pick 1. I tested the algorithm without
        # this initialization and it did learn the same curve, but I see no 
        # reason to change the initialization and I wanted to explain the numbers.
        self.open_base = torch.nn.Parameter(torch.tensor(
                                            [2.0 for x in range(42)],
                                            requires_grad=True,
                                            dtype=torch.float
                                        ))
        self.lift = torch.nn.Parameter(torch.tensor(7.0,requires_grad=True))
        #placeholder for future versions. Currently doesn't update, but I want to explore that.
        self.arch_bias = torch.ones(self.n_archs,requires_grad=False)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self,x):
        #gets pool of cards drafted so far
        pool = x[:,:self.n_cards]
        #gets current pack to decide what to pick from
        pack = x[:,self.n_cards:]
        #computes where in the draft it is. This is to index
        #into self.open_base. Clamp above 42 to enable using
        #the model as an oracle ('what if it had all the red cards').
        pick_n = torch.clamp(pool.sum(1),max=41)
        #squash open_base numbers between 0 and 1
        open_base = self.sigmoid(self.open_base)
        #enforce decaying structure over time
        open_decay = torch.cumprod(open_base,dim=0) * self.relu(self.lift)
        #enforce this decay to go to zero
        open_decay = open_decay - open_decay.min()
        #get the proper open bias forr each pick
        open_factor = open_decay[pick_n.type(torch.long)]
        #compute bias towards card drafted so far
        simple_pull = torch.matmul(pool,self.rank_matrix)
        pull_relu = self.relu(simple_pull)
        #I want to explore a version of this model where the pool informs
        #the pull as a probability distribution across archetypes, and then
        #has some memory of past cards seen to also update that distribution
        #based on what it models as open, but for now this doesn't exist
        pull_final = (pull_relu * self.arch_bias) + open_factor[:,None]
        #rank every card in the format according to bias
        pick_rankings = torch.matmul(pull_final,self.rank_matrix.t())
        pick_relu = self.relu(pick_rankings)
        #zero value for all cards that are not in the current pack
        return pick_relu * pack