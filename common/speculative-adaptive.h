#pragma once

#include <algorithm>

// Adaptive draft depth controller for MTP speculative decoding (draft-mtp-adaptive).
//
// Hysteresis state machine: the depth climbs one step after N consecutive
// full-accept verifies and drops one step when accumulated miss pressure
// (sum of n_draft - n_accepted per round) reaches a depth-scaled budget.
// The depth stays within [floor, cap]; at the floor no pressure accumulates.
struct common_speculative_adaptive {
    int n_cur   = 0; // current adaptive draft depth N
    int n_climb = 0; // consecutive verifies that accepted every drafted token
    int n_drop  = 0; // accumulated drop pressure: sum of (n_draft - n_accepted)

    static int climb_threshold(int depth) {
        switch (depth) {
            case 1: return 2;
            case 2: return 4;
            case 3: return 10; // hardened 3->4 barrier: keeps prose/reasoning pinned
            case 4: return 6;
            case 5: return 3;
            case 6: return 2;
            default: return 2; // depth >= 7
        }
    }

    static int drop_pressure(int depth) {
        return std::max(depth * 5, 20);
    }

    void reset(int n_max, int n_min_adaptive) {
        const int cap   = std::max(1, n_max);
        const int floor = std::max(1, n_min_adaptive);

        n_cur   = std::min(floor, cap);
        n_climb = 0;
        n_drop  = 0;
    }

    void update(int n_draft, int n_accepted, int n_max, int n_min_adaptive) {
        if (n_draft <= 0) {
            return;
        }

        const int cap   = std::max(1, n_max);
        const int floor = std::max(1, n_min_adaptive);

        if (n_accepted == n_draft) {
            n_drop = 0;

            if (n_cur < cap && ++n_climb >= climb_threshold(n_cur)) {
                n_cur++;
                n_climb = 0;
            }
        } else {
            n_climb = 0;

            if (n_cur > floor) {
                n_drop += n_draft - n_accepted;
                if (n_drop >= drop_pressure(n_cur)) {
                    n_cur--;
                    n_drop = 0;
                }
            }
        }
    }
};
