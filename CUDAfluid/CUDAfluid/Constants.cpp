#include "Constants.h"

const int gridXRes = 200;
const int gridYRes = 200;
const int gridZRes = 200;

const int numGridCells = gridXRes * gridYRes * gridZRes;
const int numXFaces = (gridXRes + 1) * gridYRes * gridZRes;
const int numYFaces = gridXRes * (gridYRes + 1) * gridZRes;
const int numZFaces = gridXRes * gridYRes * (gridZRes + 1);

float timestep = 0.08f;
float cellSize = 1.0f;

float fluidMass = 1.0f;
float gravity = -9.8f;