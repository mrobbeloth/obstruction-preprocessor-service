#pragma once
#include "ProjectUtilities.h"
#include "includes.h"

class Direction {	
    private:
        int value;
        enum {
            S = 0, SW = 1, W = 2, NW = 3, N = 4, NE = 5, E = 6, SE = 7
        } typedef direction;


        static const vector<string> getAllDirectionNames() {
            vector<string> dirs = {"S", "SW", "W", "NW", "N", "NE", "E", "SE"};
            return dirs;
        }

        static const vector<direction> getAllDirections() {
            static const vector<direction> allDirections = {
                direction::S,
                direction::SW,
                direction::W,
                direction::NW,
                direction::N,
                direction::NE,
                direction::E,
                direction::SE
            };
            return allDirections;
        }
    
    public:
        Direction(int val) : value(val) {};
        
        static string getEnumByString(int value) {
            string cardinalDirection;
            for (Direction::direction d : getAllDirections()) {
                if (static_cast<int>(d) == value) {
                    cardinalDirection = getAllDirectionNames()[d];
                    break;
                }
            }
            return cardinalDirection;
        }
};