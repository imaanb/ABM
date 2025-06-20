import math

from mesa.discrete_space import CellAgent


# Helper function
def get_distance(cell_1, cell_2):
    """
    Calculate the Euclidean distance between two positions

    used in trade.move()
    """

    x1, y1 = cell_1.coordinate
    x2, y2 = cell_2.coordinate
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx**2 + dy**2)


class Trader(CellAgent):
    """
    Trader:
    - has a metabolism of sugar and spice
    - harvest and trade sugar and spice to survive
    """

    def __init__(
        self,
        model,
        cell,
        sugar=0,
        spice=0,
        metabolism_sugar=0,
        metabolism_spice=0,
        vision=0,
        income_tax_sugar=0.05,
        income_tax_spice=0.04,
        strategy=None,
    ):
        super().__init__(model)
        self.cell = cell
        self.sugar = sugar
        self.spice = spice
        self.initial_sugar = sugar
        self.initial_spice = spice
        self.metabolism_sugar = metabolism_sugar
        self.metabolism_spice = metabolism_spice
        self.vision = vision
        self.prices = []
        self.trade_partners = []
        self.income_tax_sugar = income_tax_sugar
        self.income_tax_spice = income_tax_spice
        self.strategy = strategy or self.random.choice(["stag", "hare"])
    
    @property
    def wealth(self):
        return self.sugar + self.spice

    def _get_income_tax_rate(self, income):
        sys = self.model.income_tax_system
        if sys == "none":
            return 0.0
        if sys == "proportional":
            return self.model.income_tax_flat_rate
        # progressive or degressive
        brackets = list(self.model.income_tax_brackets)
        if sys == "degressive":
            # invert only the rates
            th, rt = zip(*brackets)
            brackets = list(zip(th, reversed(rt)))
        for upper, rate in brackets:
            if income <= upper:
                return rate
        return 0.0

    def play_staghunt(self):
        """
        Pair up with one random neighbor Trader (within vision) and
        play a stag‐hunt.  Returns the sugars earned by *this* agent,
        and deposits the partner’s payoff into their own sugar store.
        """

        neighbors = []
        for cell in self.cell.get_neighborhood(self.vision, include_center=False):
            for ag in cell.agents:
                if isinstance(ag, Trader) and ag is not self:
                    neighbors.append(ag)

        if not neighbors:
            return 0

        other = self.random.choice(neighbors)

        payoff_you, payoff_them = self.model.staghunt_payoffs[
            (self.strategy, other.strategy)
        ]

        other.sugar += payoff_them

        return payoff_you

    def get_trader(self, cell):
        """
        helper function used in self.trade_with_neighbors()
        """

        for agent in cell.agents:
            if isinstance(agent, Trader):
                return agent

    def calculate_welfare(self, sugar, spice):
        """
        helper function

        part 2 self.move()
        self.trade()
        """

        # calculate total resources
        m_total = self.metabolism_sugar + self.metabolism_spice
        # Cobb-Douglas functional form; starting on p. 97
        # on Growing Artificial Societies
        return sugar ** (self.metabolism_sugar / m_total) * spice ** (
            self.metabolism_spice / m_total
        )

    def is_starved(self):
        """
        Helper function for self.maybe_die()
        """


        return (self.sugar <= 0) or (self.spice <= 0)

    def calculate_MRS(self, sugar, spice):
        """
        Helper function for
          - self.trade()
          - self.maybe_self_spice()

        Determines what trader agent needs and can give up
        """

        return (spice / self.metabolism_spice) / (sugar / self.metabolism_sugar)

    def calculate_sell_spice_amount(self, price):
        """
        helper function for self.maybe_sell_spice() which is called from
        self.trade()
        """

        if price >= 1:
            sugar = 1
            spice = int(price)
        else:
            sugar = int(1 / price)
            spice = 1
        return sugar, spice

    def sell_spice(self, other, sugar, spice):
        """
        used in self.maybe_sell_spice()

        exchanges sugar and spice between traders
        """
        vat_rate_sugar, vat_rate_spice  = self.model.vat_rate_sugar, self.model.vat_rate_spice

        sugar_vat = sugar * vat_rate_sugar
        spice_vat = spice * vat_rate_spice


        self.sugar += sugar -sugar_vat
        other.sugar -= sugar
        self.spice -= spice
        other.spice += spice -spice_vat

        self.model.treasury["sugar"] += sugar_vat 
        self.model.treasury["spice"] += spice_vat


    def maybe_sell_spice(self, other, price, welfare_self, welfare_other):
        """
        helper function for self.trade()
        """

        sugar_exchanged, spice_exchanged = self.calculate_sell_spice_amount(price)
        
        
        
        vat_rate_sugar, vat_rate_spice = self.model.vat_rate_sugar, self.model.vat_rate_spice 
        sugar_vat = sugar_exchanged * vat_rate_sugar
        spice_vat = spice_exchanged * vat_rate_spice


        # Assess new sugar and spice amount - what if change did occur
        self_sugar = self.sugar + sugar_exchanged - sugar_vat
        other_sugar = other.sugar - sugar_exchanged
        self_spice = self.spice - spice_exchanged
        other_spice = other.spice + spice_exchanged -spice_vat
        # double check to ensure agents have resources

        if (
            (self_sugar <= 0)
            or (other_sugar <= 0)
            or (self_spice <= 0)
            or (other_spice <= 0)
        ):
            return False

        # trade criteria #1 - are both agents better off?
        both_agents_better_off = (
            welfare_self < self.calculate_welfare(self_sugar, self_spice)
        ) and (welfare_other < other.calculate_welfare(other_sugar, other_spice))

        # trade criteria #2 is their mrs crossing with potential trade
        mrs_not_crossing = self.calculate_MRS(
            self_sugar, self_spice
        ) > other.calculate_MRS(other_sugar, other_spice)

        if not (both_agents_better_off and mrs_not_crossing):
            return False

        # criteria met, execute trade
        self.sell_spice(other, sugar_exchanged, spice_exchanged)
        return True

    def trade(self, other):
        """
        helper function used in trade_with_neighbors()

        other is a trader agent object
        """

        # sanity check to verify code is working as expected
        assert self.sugar > 0
        assert self.spice > 0
        assert other.sugar > 0
        assert other.spice > 0

        # calculate marginal rate of substitution in Growing Artificial Societies p. 101
        mrs_self = self.calculate_MRS(self.sugar, self.spice)
        mrs_other = other.calculate_MRS(other.sugar, other.spice)

        # calculate each agents welfare
        welfare_self = self.calculate_welfare(self.sugar, self.spice)
        welfare_other = other.calculate_welfare(other.sugar, other.spice)

        if math.isclose(mrs_self, mrs_other):
            return

        # calculate price
        price = math.sqrt(mrs_self * mrs_other)

        

        if mrs_self > mrs_other:
            # self is a sugar buyer, spice seller
            sold = self.maybe_sell_spice(other, price, welfare_self, welfare_other)
            # no trade - criteria not met
            if not sold:
                return
            
        else:
            # self is a spice buyer, sugar seller
            sold = other.maybe_sell_spice(self, price, welfare_other, welfare_self)
            # no trade - criteria not met
            if not sold:
                return

        # Capture data
        self.prices.append(price)
        self.trade_partners.append(other.unique_id)
        self.model.trades_made += 1 
        # continue trading
        self.trade(other)

    ######################################################################
    #                                                                    #
    #                      MAIN TRADE FUNCTIONS                          #
    #                                                                    #
    ######################################################################

    def move(self):
        """
        Function for trader agent to identify optimal move for each step in 4 parts
        1 - identify all possible moves
        2 - determine which move maximizes welfare
        3 - find closest best option
        4 - move
        """

        # 1. identify all possible moves

        neighboring_cells = [
            cell
            for cell in self.cell.get_neighborhood(self.vision, include_center=True)
            if cell.is_empty
        ]

        if not neighboring_cells:
            # no empty neighboring cells, stay put
            return
        # 2. determine which move maximizes welfare
        
        welfares = [
            self.calculate_welfare(
                self.sugar + cell.sugar,
                self.spice + cell.spice,
            )
            for cell in neighboring_cells
        ]

        # 3. Find closest best option

        # find the highest welfare in welfares
        max_welfare = max(welfares)
        # get the index of max welfare cells
        # fixme: rewrite using enumerate and single loop
        candidate_indices = [
            i for i in range(len(welfares)) if math.isclose(welfares[i], max_welfare)
        ]

        # convert index to positions of those cells
        candidates = [neighboring_cells[i] for i in candidate_indices]

        min_dist = min(get_distance(self.cell, cell) for cell in candidates)

        final_candidates = [
            cell
            for cell in candidates
            if math.isclose(get_distance(self.cell, cell), min_dist, rel_tol=1e-02)
        ]

        # 4. Move Agent
        self.cell = self.random.choice(final_candidates)

    def eat(self):
        base_sugar = self.cell.sugar
        base_spice = self.cell.spice

        bonus_sugar = 0
        if self.model.enable_staghunt:
            bonus_sugar = self.play_staghunt()

        income_sugar = base_sugar + bonus_sugar
        income_spice = base_spice

        self._last_income_sugar = income_sugar
        self._last_income_spice = income_spice

        rate_s = self._get_income_tax_rate(income_sugar)
        rate_p = self._get_income_tax_rate(income_spice)
        tax_s = rate_s * income_sugar
        tax_p = rate_p * income_spice

        self.model.treasury["sugar"]  += tax_s
        self.model.treasury["spice"]  += tax_p

        self.sugar += income_sugar - tax_s
        self.spice += income_spice - tax_p

        self.cell.sugar = 0
        self.cell.spice = 0

        self.sugar -= self.metabolism_sugar
        self.spice -= self.metabolism_spice

    def maybe_die(self):
        """
        Function to remove Traders who have consumed all their sugar or spice
        """

        if self.is_starved():
            self.sugar = self.initial_sugar
            self.spice = self.initial_spice
            self.cell = self.model.random.choice(self.model.grid.all_cells.cells)




    def trade_with_neighbors(self):
        """
        Function for trader agents to decide who to trade with in three parts

        1- identify neighbors who can trade
        2- trade (2 sessions)
        3- collect data
        """
        # iterate through traders in neighboring cells and trade
        for a in self.cell.get_neighborhood(radius=self.vision).agents:
            self.trade(a)

        return

    def pay_wealth_tax(self):
        wealth = self.sugar + self.spice
        sys = self.model.wealth_tax_system

        if sys == "none":
            rate = 0.0
        elif sys == "proportional":
            rate = self.model.flat_rate
        else:
            # pick the bracket rate
            brackets = self.model.wealth_tax_brackets
            # if degressive, invert the order of rates
            if sys == "degressive":
                brackets = [
                    (ub, r)
                    for (ub, r) in zip(
                        [b[0] for b in brackets], reversed([b[1] for b in brackets])
                    )
                ]
            rate = next(r for ub, r in brackets if wealth <= ub)

        # now collect the tax
        sugar_tax = rate * self.sugar
        spice_tax = rate * self.spice

        self.sugar -= sugar_tax
        self.spice -= spice_tax

        self.model.treasury["sugar"] += sugar_tax
        self.model.treasury["spice"] += spice_tax
        

