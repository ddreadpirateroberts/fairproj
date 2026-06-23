from fairproj.bases import (
    Piece, Allocation, Agent, 
    ProportionalProtocol, 
    EnvyFreeProtocol, 
    TwoAgentProtocol, 
    ThreeAgentProtocol, 
    StromquistConvergenceError,
    E
)
from typing import List, Tuple, Union


class CutAndChoose(TwoAgentProtocol, EnvyFreeProtocol):
    def run(self) -> Allocation:
        cutter, chooser = self.agents
        alloc = Allocation()
        
        cut = cutter.mark(target_value=0.5)
        
        left = Piece(0, cut)
        right = Piece(cut, 1)
        
        if chooser.eval(left) < chooser.eval(right): 
            alloc.assign(chooser, right)
            alloc.assign(cutter, left)
        else: 
            alloc.assign(chooser, left)
            alloc.assign(cutter, right)

        return alloc 


class SteinhausProcedure(ThreeAgentProtocol, ProportionalProtocol):
    def run(self) -> Allocation:
        v1, v2, v3 = self.agents
        alloc = Allocation()
        
        cut1 = v1.mark(target_value=1/3)
        cut2 = v1.mark(target_value=1/3, start=cut1)

        pieces = [
            Piece(0, cut1), 
            Piece(cut1, cut2), 
            Piece(cut2, 1)
        ]
        
        acceptable_count = {v2: 0, v3: 0}
        for p in pieces: 
            if v2.eval(p) >= 1/3 - E: 
                acceptable_count[v2] += 1 
            if v3.eval(p) >= 1/3 - E:
                acceptable_count[v3] += 1 
        
        if acceptable_count[v2] >= 2: 
            # V3 -> V2 -> V1
            v3_pick, remainder = v3.pick_from(pieces)
            v2_pick, remainder = v2.pick_from(remainder)
            v1_pick = remainder
            
        elif acceptable_count[v3] >= 2: 
            # V2 -> V3 -> V1
            v2_pick, remainder = v2.pick_from(pieces)
            v3_pick, remainder = v3.pick_from(remainder)
            v1_pick = remainder
        else: 
            # V1 gets the 'bad piece' in V2 and V3's book. 
            # V2 and V3 play cut and choose. 
            bad_piece = [
                p for p in pieces 
                if v2.eval(p) < 1/3-E and v3.eval(p) < 1/3-E
            ][0]
            
            v1_pick = [bad_piece]
            remainder = [p for p in pieces if p != bad_piece]
            
            # V2 cuts and V3 chooses 
            r1, r2 = remainder
            tv2 = v2.eval(r1) + v2.eval(r2)
        
            if v2.eval(r1) >= tv2 / 2:
                cut3 = v2.mark(tv2 / 2, start=r1.left, end=r1.right)
                half_a = [Piece(r1.left, cut3)]
                half_b = [Piece(cut3, r1.right), r2]
            else:
                cut3 = v2.mark(tv2 / 2 - v2.eval(r1), start=r2.left, end=r2.right)
                half_a = [r1, Piece(r2.left, cut3)]
                half_b = [Piece(cut3, r2.right)]

            if sum(v3.eval(p) for p in half_a) >= sum(v3.eval(p) for p in half_b):
                v3_pick = half_a
                v2_pick = half_b
            else:
                v3_pick = half_b
                v2_pick = half_a
                
        assignments = {
            v1: v1_pick, 
            v2: v2_pick, 
            v3: v3_pick
        }
        alloc.assign(assignments)

        return alloc 

        
class LastDiminisher(ProportionalProtocol):
    """Banach-Knaster. Non-contiguous. Works for any n."""            
    def run(self) -> Allocation:
        alloc = Allocation()
        playing_agents = self.agents[:]
        
        left = 0. 
        while len(playing_agents) > 1: 
            right = 1.
            n = len(playing_agents)
            current_piece = Piece(left, right)
                 
            diminishers = []
            trimmed = current_piece
            for agent in playing_agents: 
                tot_val = agent.eval(current_piece)
                if agent.eval(trimmed) >= tot_val / n - E: 
                    right = agent.mark(tot_val/n, start=left, end=right)
                    trimmed = Piece(left, right)
                    diminishers.append(agent)
                    
            last_dim = diminishers[-1]
            alloc.assign(last_dim, trimmed)
            left = right
            playing_agents.remove(last_dim)
        
        last_agent = playing_agents[0]
        remaining_cake = Piece(left, 1)
        alloc.assign(last_agent, remaining_cake)
        
        return alloc 


class DubinsSpanier(ProportionalProtocol): 
    """The Discrete version. Contiguous. Works for any n."""
    def _procedure(self, active_agents: List[Agent], 
                   alloc: Allocation, rest: Piece) -> Allocation: 
        n = len(active_agents)
                    
        if n == 1: 
            alloc.assign(active_agents[0], rest)
            return alloc 
        
        marks = []
        for a in active_agents:
            mark_at = a.eval(rest) / n
            mark = a.mark(mark_at, start=rest.left, end=rest.right)
            marks.append((a, mark))

        marks.sort(key=lambda tpl: tpl[1])

        shouter, shouter_mark = marks[0]
        _, second_mark  = marks[1]

        cut = (shouter_mark + 10 * second_mark) / 11
        
        alloc.assign(shouter, Piece(rest.left, cut))
        active_agents.remove(shouter)
        
        return self._procedure(active_agents, alloc, Piece(cut, rest.right))
    
    def run(self) -> Allocation: 
        alloc = Allocation()
        return self._procedure(self.agents[:], alloc, Piece(0, 1))


class EvenPaz(ProportionalProtocol):
    """Divide-and-conquer. Any n, contiguous, optimal query count."""
    def _procedure(self, piece: Piece, agents: List[Agent], 
                   alloc: Allocation) -> Allocation: 
        n = len(agents)        
        
        if n == 1: 
            alloc.assign(agents[0], piece)
            return alloc 
        
        marks = []
        target = ((n // 2) / n)  
        for a in agents: 
            mark_at = target * a.eval(piece)
            marks.append((a, a.mark(mark_at, piece.left, piece.right)))
        
        marks.sort(key=lambda tpl: tpl[1])
        
        div = n // 2
        left_agents = [t[0] for t in marks[:div]]
        right_agents = [t[0] for t in marks[div:]]
        cut = (marks[div-1][1] + marks[div][1]) / 2
        
        self._procedure(Piece(piece.left, cut), left_agents, alloc)
        self._procedure(Piece(cut, piece.right), right_agents, alloc)
        
        return alloc 
    
    def run(self) -> Allocation:
        alloc = Allocation()
        return self._procedure(Piece(0, 1), self.agents, alloc)
        

class SelfridgeConway(ThreeAgentProtocol, EnvyFreeProtocol):    
    def run(self) -> Allocation:
        alloc = Allocation()
        v1, v2, v3 = self.agents

        cut1, cut2 = v1.mark(1/3), v1.mark(2/3)
        pieces = [Piece(0, cut1), Piece(cut1, cut2), Piece(cut2, 1)]

        worst, second_best, best = sorted(pieces, key=lambda p: v2.eval(p))

        trim_point = v2.mark(v2.eval(second_best), best.left, best.right)
        trimmed = Piece(best.left, trim_point)
        trimmings = Piece(trim_point, best.right)

        choosable = [worst, second_best, trimmed]

        v3_pick, remaining = v3.pick_from(choosable)

        if v3_pick[0] != trimmed:
            v2_pick = [trimmed]
            v1_pick = [p for p in remaining if p != trimmed]
            cutter, noncutter = v3, v2
        else:
            v2_pick, remaining = v2.pick_from(remaining)
            v1_pick = remaining
            cutter, noncutter = v2, v3

        alloc.assign({v1: v1_pick, v2: v2_pick, v3: v3_pick})

        t = cutter.eval(trimmings) / 3
        cut3 = cutter.mark(t, trimmings.left, trimmings.right)
        cut4 = cutter.mark(t*2, trimmings.left, trimmings.right)

        trimming_pieces = [
            Piece(trimmings.left, cut3),
            Piece(cut3, cut4),
            Piece(cut4, trimmings.right),
        ]

        noncutter_pick, remaining = noncutter.pick_from(trimming_pieces)
        v1_trimming_pick, remaining = v1.pick_from(remaining)
        cutter_pick = remaining

        alloc.assign({
            noncutter: noncutter_pick,
            v1: v1_trimming_pick,
            cutter: cutter_pick,
        })

        return alloc
    
class Stromquist(ThreeAgentProtocol, EnvyFreeProtocol):
    def _who_shouts(
        self, 
        refs_piece: Piece, 
        sorted_marks: List[Tuple[Agent, float]]
    ) -> Union[Tuple[Agent, Agent, Agent, List[Piece]], None]: 
        leftmost, central, rightmost = sorted_marks 
        pieces = [
            refs_piece, 
            Piece(refs_piece.right, central[1]), 
            Piece(central[1], 1)
        ]
        
        # Central calls stop when the three 
        # pieces would be equal to her
        cEval = central[0].eval
        if abs(cEval(pieces[0]) - cEval(pieces[1])) < E: 
            if abs(cEval(pieces[0]) - cEval(pieces[2])) < E: 
                return (
                    central[0], leftmost[0], 
                    rightmost[0], pieces
                )
        
        # rightmost calls stop when the right 
        # part of the cake equals the left part
        rEval = rightmost[0].eval
        if abs(rEval(pieces[0]) - rEval(pieces[2])) < E: 
            return (
                rightmost[0], leftmost[0],
                central[0], pieces      
            )
        
        # leftmost calls stop when the middle 
        # part of the cake equals the left part
        lEval = leftmost[0].eval
        if abs(lEval(pieces[0]) - lEval(pieces[1])) < E: 
            return (
                leftmost[0], central[0], 
                rightmost[0], pieces
            )
            
        return None 

    def run(self, delta=1e-5) -> Allocation:
        alloc = Allocation()
        
        refs_knife = 0
        while True: 
            refs_knife += delta
            if refs_knife >= 1: 
                raise StromquistConvergenceError(delta)
                
            refrees_cut = Piece(0, refs_knife)
            right_piece = Piece(refs_knife, 1)
            
            marks = []
            for agent in self.agents: 
                mark = agent.mark(
                    agent.eval(right_piece) / 2, 
                    start=right_piece.left, 
                    end=right_piece.right
                )
                marks.append((agent, mark))
                
            marks.sort(key=lambda tpl: tpl[1])
            
            anyone = self._who_shouts(refrees_cut, marks)
            if anyone: 
                shouter, left, right, pieces = anyone
                if shouter: 
                    alloc.assign({
                        shouter: pieces[:1], 
                        left: pieces[1:2], 
                        right: pieces[2:]
                    })
                    return alloc 

             
if __name__ == "__main__": 
    print("hi")
