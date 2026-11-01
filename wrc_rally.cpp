// ============================================================================
//  WRC RALLY1 HYBRID SIMULATION  -  C++17 / OpenGL (fixed function) / X11
//  Build:  g++ -O3 -std=c++17 wrc_rally.cpp -o wrc_rally -lX11 -lGL -lGLU -lm
//
//  Controls (drive):  W/S throttle-brake(reverse)  A/D steer  SPACE handbrake
//                     C camera  V next car  R reset to road  K new climate/scenery
//                     M back to 2D map  P autopilot demo  ESC quit
//  Controls (draw):   LMB drag draw  L close circuit  C / RMB clear  ENTER drive
//  Joystick:          /dev/input/js0  (env WRC_AXIS_STEER / _THROTTLE / _BRAKE)
// ============================================================================

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>

#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <linux/joystick.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <random>
#include <string>
#include <vector>

#ifndef GL_MULTISAMPLE
#define GL_MULTISAMPLE 0x809D
#endif
#ifndef GL_TEXTURE_MAX_ANISOTROPY_EXT
#define GL_TEXTURE_MAX_ANISOTROPY_EXT 0x84FE
#endif
#ifndef GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT
#define GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT 0x84FF
#endif
#ifndef GL_LIGHT_MODEL_COLOR_CONTROL
#define GL_LIGHT_MODEL_COLOR_CONTROL 0x81F8
#endif
#ifndef GL_SEPARATE_SPECULAR_COLOR
#define GL_SEPARATE_SPECULAR_COLOR 0x81FA
#endif
#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif

// ============================================================================
//  MATH
// ============================================================================
static constexpr float PI = 3.14159265358979323846f;
static constexpr float RAD2DEG = 180.0f / PI;

struct Vec2 {
    float x = 0, y = 0;
    Vec2() = default;
    Vec2(float _x, float _y) : x(_x), y(_y) {}
};

struct Vec3 {
    float x = 0, y = 0, z = 0;
    Vec3() = default;
    Vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
    Vec3 operator+(const Vec3& o) const { return Vec3(x + o.x, y + o.y, z + o.z); }
    Vec3 operator-(const Vec3& o) const { return Vec3(x - o.x, y - o.y, z - o.z); }
    Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
    Vec3 operator-() const { return Vec3(-x, -y, -z); }
    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    float length() const { return std::sqrt(x * x + y * y + z * z); }
    float lengthXZ() const { return std::sqrt(x * x + z * z); }
    Vec3 normalized() const {
        float l = length();
        return l > 1e-6f ? Vec3(x / l, y / l, z / l) : Vec3(0, 1, 0);
    }
};
static inline float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static inline float clampf(float v, float lo, float hi) { return std::max(lo, std::min(hi, v)); }
static inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }
static inline float smooth01(float t) { t = clampf(t, 0.0f, 1.0f); return t * t * (3.0f - 2.0f * t); }
static inline float wrapPi(float a) {
    while (a > PI) a -= 2.0f * PI;
    while (a < -PI) a += 2.0f * PI;
    return a;
}
static inline float sgnf(float v) { return v >= 0.0f ? 1.0f : -1.0f; }

// ============================================================================
//  CONSTANTS
// ============================================================================
static constexpr float STAGE_STEP = 3.5f;       // metres between stage nodes
static constexpr float CURB_W = 0.9f;           // red/white curb width
static constexpr float SHOULDER_W = 16.0f;      // sloped bank width
static constexpr float TERRAIN_HALF = 1800.0f;  // terrain half extent (m)
static constexpr float TERRAIN_STEP = 10.0f;    // heightfield resolution (m)
static constexpr int   HF_N = (int)(2 * TERRAIN_HALF / TERRAIN_STEP) + 1;
static constexpr float GRAV = 9.81f;
static constexpr float PHYS_DT = 1.0f / 240.0f;

// ============================================================================
//  DATA TYPES
// ============================================================================
enum AppState { STATE_DRAW_MAP = 0, STATE_RALLY_DRIVE = 1 };

enum ClimateType {
    CLIMATE_SUNNY_GRAVEL = 0,    // Gunesli Toprak Etabi        (%100)
    CLIMATE_HEAVY_RAIN_MUD = 1,  // Saganak Yagmur & Camur      (%68)
    CLIMATE_BLIZZARD_SNOW = 2,   // Kar Firtinasi & Buzlu Etap  (%52)
    CLIMATE_DENSE_FOG_DUSK = 3   // Yogun Sis & Kizil Gunbatimi (%85)
};

struct ClimateProfile {
    const char* name;
    float surfaceGrip;
    float skyTop[3], skyHorizon[3];
    float fogColor[4];
    float fogDensity;
    float sunDir[3];
    float sunColor[4], ambient[4], fill[4];
    float ground[3], rock[3], roadTint[3], foliage[3], dust[3];
    bool snow, rain;
    float cloudCover;
    float cloudColor[3];
    float sunGlow;
};

static const ClimateProfile CLIMATES[4] = {
    { "GUNESLI TOPRAK ETABI", 1.00f,
      {0.16f, 0.40f, 0.82f}, {0.72f, 0.83f, 0.95f}, {0.72f, 0.83f, 0.95f, 1.0f}, 0.00115f,
      {0.45f, 0.62f, 0.40f}, {1.00f, 0.95f, 0.82f, 1.0f}, {0.34f, 0.38f, 0.46f, 1.0f}, {0.20f, 0.24f, 0.32f, 1.0f},
      {0.50f, 0.62f, 0.32f}, {0.58f, 0.55f, 0.50f}, {1.00f, 0.88f, 0.70f}, {0.24f, 0.46f, 0.17f}, {0.80f, 0.68f, 0.52f},
      false, false, 0.35f, {1.0f, 1.0f, 1.0f}, 1.0f },
    { "SAGANAK YAGMUR & CAMUR ETABI", 0.68f,
      {0.17f, 0.20f, 0.25f}, {0.45f, 0.49f, 0.54f}, {0.45f, 0.49f, 0.54f, 1.0f}, 0.0030f,
      {0.25f, 0.70f, -0.35f}, {0.58f, 0.61f, 0.67f, 1.0f}, {0.30f, 0.33f, 0.38f, 1.0f}, {0.16f, 0.18f, 0.22f, 1.0f},
      {0.30f, 0.40f, 0.22f}, {0.40f, 0.40f, 0.40f}, {0.60f, 0.48f, 0.38f}, {0.15f, 0.30f, 0.13f}, {0.34f, 0.26f, 0.18f},
      false, true, 0.95f, {0.42f, 0.44f, 0.48f}, 0.0f },
    { "KAR FIRTINASI & BUZLU ETAP", 0.52f,
      {0.58f, 0.64f, 0.74f}, {0.84f, 0.88f, 0.93f}, {0.84f, 0.88f, 0.93f, 1.0f}, 0.0038f,
      {-0.40f, 0.65f, 0.45f}, {0.82f, 0.86f, 0.95f, 1.0f}, {0.50f, 0.54f, 0.62f, 1.0f}, {0.30f, 0.34f, 0.40f, 1.0f},
      {0.93f, 0.95f, 1.00f}, {0.60f, 0.62f, 0.66f}, {0.86f, 0.90f, 0.96f}, {0.17f, 0.32f, 0.20f}, {0.92f, 0.95f, 1.00f},
      true, false, 0.90f, {0.82f, 0.85f, 0.90f}, 0.0f },
    { "YOGUN SIS & KIZIL GUNBATIMI", 0.85f,
      {0.14f, 0.09f, 0.26f}, {0.94f, 0.52f, 0.30f}, {0.80f, 0.47f, 0.36f, 1.0f}, 0.0036f,
      {-0.78f, 0.20f, -0.52f}, {1.00f, 0.52f, 0.25f, 1.0f}, {0.32f, 0.23f, 0.30f, 1.0f}, {0.20f, 0.14f, 0.24f, 1.0f},
      {0.38f, 0.38f, 0.22f}, {0.50f, 0.42f, 0.40f}, {0.86f, 0.62f, 0.50f}, {0.20f, 0.28f, 0.13f}, {0.70f, 0.52f, 0.42f},
      false, false, 0.55f, {1.0f, 0.62f, 0.45f}, 2.2f }
};

struct StageNode {
    Vec3 pos;          // 3D centre line position
    Vec3 forward;      // unit tangent
    Vec3 right;        // unit lateral normal (visual right-hand side of the road)
    Vec3 up;           // surface normal
    float curvature;   // signed (+ = left turn) [1/m]
    float slope;       // dy/ds along road
    float dist;        // distance from stage start
    float roadWidth;
};

enum EventKind { EVT_CORNER = 0, EVT_JUMP = 1 };
struct StageEvent {          // pace note
    int kind;
    int dir;                 // +1 left, -1 right, 0 jump
    int sev;                 // 1 = HP(hairpin), 2, 3, 4 = JP
    int startIdx, endIdx;
};

enum SceneryKind {
    SCN_CONIFER = 0, SCN_BROADLEAF, SCN_BUSH, SCN_ROCK, SCN_HAYBALE,
    SCN_BUILDING, SCN_GRANDSTAND, SCN_ARCH, SCN_CHEVRON
};

struct SceneryObject {
    SceneryKind kind = SCN_CONIFER;
    Vec3 pos;
    float yaw = 0;           // degrees
    float scale = 1;
    float w = 1, d = 1, h = 1;
    int variant = 0;
    float color[3] = {1, 1, 1};
    int solid = 0;           // 0 none, 1 circle, 2 oriented box
    float radius = 0;
    int archType = 0;        // 0 start, 1 finish
};

struct Particle3D {
    Vec3 pos, vel;
    float color[4];
    float size, growth;
    float life, maxLife;
    int kind;                // 1 dust/spray, 2 spark
};

struct SkidMark3D {
    Vec3 a, b;               // segment start/end (ground contact)
    float alpha;
};

struct RallyCar {
    // pose
    Vec3 pos;                // y = ground contact height under centre of mass
    float heading = 0;       // rad, 0 = +Z, increasing = turning left
    float vx = 0, vy = 0;    // body-frame velocity (x fwd, y left)
    float yawRate = 0;
    float speed = 0;         // = vx
    float lateralSpeed = 0;  // = vy
    float vertVel = 0;       // airborne vertical velocity
    // suspension / attitude
    float suspBounce = 0, suspVel = 0;
    float bodyRoll = 0, bodyRollVel = 0;
    float bodyPitch = 0, bodyPitchVel = 0;
    float attPitch = 0, attRoll = 0;     // geometric attitude (road / air)
    float groundVy = 0, prevGroundVy = 0, groundAccel = 0, prevGroundY = 0;
    bool airborne = false;
    float airTime = 0;
    // powertrain
    float rpm = 1200;
    int gear = 1;            // 0 = R, 1..6
    float shiftTimer = 0;
    float hybridEnergy = 0.35f;
    bool hybridActive = false, hybridReady = true;
    float wheelSpin = 0;     // slip indicator 0..1
    // controls (post smoothing)
    float throttle = 0, brake = 0, steerInput = 0, steerAngle = 0;
    float handbrake = 0;
    float wheelRot = 0;      // rad
    float ax = 0, ay = 0;    // body accelerations (m/s^2)
    // telemetry
    float driftScore = 0, driftCombo = 0, slipAngle = 0;
    float stageTime = 0;
    int nearestNode = 0;
    int model = 0;
    float hitCooldown = 0;
    // visual realism & dynamics
    float brakeHeat = 0;         // 0..1 brake rotor glow temperature
    float backfireTimer = 0;     // anti-lag exhaust flame duration
    float backfireIntensity = 0; // flame brightness & size
    float wiperAngle = 0;        // windshield wiper sweep
    float prevThrottle = 0;      // detect sudden throttle lift
    // skid tracking per wheel (FL FR RL RR)
    Vec3 lastSkid[4];
    bool hasSkid[4] = {false, false, false, false};
};

struct RoadQuery {
    int idx = 0;             // segment start node
    float t = 0;             // 0..1 within segment
    float lateral = 0;       // + to the right
    float roadY = 0;
    float halfW = 4.5f;
    float distSq = 0;
    Vec3 closest;
    Vec3 right;
    Vec3 fwd;
};

// ============================================================================
//  GLOBAL STATE
// ============================================================================
static Display* g_dpy = nullptr;
static Window g_win = 0;
static GLXContext g_ctx = nullptr;
static int g_winW = 1366, g_winH = 768;
static bool g_running = true;

static AppState g_appState = STATE_DRAW_MAP;
static std::mt19937 g_rng(1337u);

// 2D map editor
static std::vector<Vec2> g_rawPts;
static bool g_drawing = false, g_closed = false;
static Vec2 g_mouse(0, 0);
static bool g_previewDirty = true;
static std::vector<Vec2> g_previewLine;   // in canvas pixels
static float g_previewScale = 1.0f;       // metres per pixel
static float g_previewLen = 0.0f, g_previewMinR = 0.0f;

// stage data
static std::vector<StageNode> g_nodes;
static std::vector<StageEvent> g_events;
static std::vector<SceneryObject> g_scenery;
static float g_stageLength = 0;
static int g_climate = 0;
static float g_surfaceGrip = 1.0f;

// node spatial grid
static constexpr float NODE_CELL = 32.0f;
static constexpr int NODE_GRID_N = (int)(2 * TERRAIN_HALF / NODE_CELL) + 1;
static std::vector<std::vector<int>> g_nodeGrid;

// height field
static std::vector<float> g_hf;          // heights
static std::vector<float> g_hfRoadDist;  // distance to road (capped)

// solid scenery grid for collisions
static constexpr float SOLID_CELL = 24.0f;
static constexpr int SOLID_GRID_N = (int)(2 * TERRAIN_HALF / SOLID_CELL) + 1;
static std::vector<std::vector<int>> g_solidGrid;

// GL resources
static GLuint g_texRoad, g_texCurb, g_texGround, g_texFacadeA, g_texFacadeB, g_texRoof, g_texWood,
    g_texPlaster, g_texRock, g_texBark, g_texNeedle, g_texSoft, g_texShadow, g_texCloud, g_texTire;
static float g_maxAniso = 1.0f;
static GLuint g_listSky = 0, g_listTerrain = 0, g_listRoad = 0;
struct SceneChunk { Vec3 center; float radius; GLuint decals; GLuint solids; };
static std::vector<SceneChunk> g_chunks;
static GLuint g_listCarBody[3] = {0, 0, 0};
static GLuint g_listWheel = 0;
static GLuint g_listBrakeStationary = 0;

// vehicle / camera
static RallyCar g_car;
static int g_cameraMode = 0;  // 0 chase, 1 low, 2 hood, 3 heli
static Vec3 g_camPos(0, 5, -10), g_camTarget(0, 1, 0);
static float g_camYaw = 0, g_shake = 0;
static float g_countdown = 3.0f, g_goFlash = 0.0f;
static bool g_finished = false;
static float g_finishTime = 0;
static int g_lap = 1, g_prevNodeIdx = 0;
static float g_lapStart = 0, g_bestLap = 0;
static float g_lightning = 0, g_lightningTimer = 8.0f;
static float g_timeAccum = 0.0f;

static std::vector<Particle3D> g_weather, g_fx;
static std::vector<SkidMark3D> g_skids;
static size_t g_skidHead = 0;
static constexpr size_t MAX_SKIDS = 4000;

// input
static bool g_keys[65536];
static bool g_autopilot = false;
struct JoyState {
    int fd = -1;
    float axis[16] = {};
    float rest[16] = {};
    bool haveRest[16] = {};
    bool handbrake = false;
    int axisSteer = 0, axisThrottle = -1, axisBrake = -1;
    float steer = 0, throttle = 0, brake = 0;
    int retry = 0;
    bool active = false;     // any joystick input seen
};
static JoyState g_joy;

// ============================================================================
//  BITMAP FONT (5x7, no X11 fonts needed) + 2D HELPERS
// ============================================================================
struct Glyph { char c; uint8_t r[7]; };
static const Glyph FONT_TABLE[] = {
    {'A',{0x0E,0x11,0x11,0x1F,0x11,0x11,0x11}}, {'B',{0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E}},
    {'C',{0x0E,0x11,0x10,0x10,0x10,0x11,0x0E}}, {'D',{0x1E,0x11,0x11,0x11,0x11,0x11,0x1E}},
    {'E',{0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F}}, {'F',{0x1F,0x10,0x10,0x1E,0x10,0x10,0x10}},
    {'G',{0x0E,0x11,0x10,0x17,0x11,0x11,0x0F}}, {'H',{0x11,0x11,0x11,0x1F,0x11,0x11,0x11}},
    {'I',{0x0E,0x04,0x04,0x04,0x04,0x04,0x0E}}, {'J',{0x07,0x02,0x02,0x02,0x02,0x12,0x0C}},
    {'K',{0x11,0x12,0x14,0x18,0x14,0x12,0x11}}, {'L',{0x10,0x10,0x10,0x10,0x10,0x10,0x1F}},
    {'M',{0x11,0x1B,0x15,0x15,0x11,0x11,0x11}}, {'N',{0x11,0x11,0x19,0x15,0x13,0x11,0x11}},
    {'O',{0x0E,0x11,0x11,0x11,0x11,0x11,0x0E}}, {'P',{0x1E,0x11,0x11,0x1E,0x10,0x10,0x10}},
    {'Q',{0x0E,0x11,0x11,0x11,0x15,0x12,0x0D}}, {'R',{0x1E,0x11,0x11,0x1E,0x14,0x12,0x11}},
    {'S',{0x0F,0x10,0x10,0x0E,0x01,0x01,0x1E}}, {'T',{0x1F,0x04,0x04,0x04,0x04,0x04,0x04}},
    {'U',{0x11,0x11,0x11,0x11,0x11,0x11,0x0E}}, {'V',{0x11,0x11,0x11,0x11,0x11,0x0A,0x04}},
    {'W',{0x11,0x11,0x11,0x15,0x15,0x15,0x0A}}, {'X',{0x11,0x11,0x0A,0x04,0x0A,0x11,0x11}},
    {'Y',{0x11,0x11,0x11,0x0A,0x04,0x04,0x04}}, {'Z',{0x1F,0x01,0x02,0x04,0x08,0x10,0x1F}},
    {'0',{0x0E,0x11,0x13,0x15,0x19,0x11,0x0E}}, {'1',{0x04,0x0C,0x04,0x04,0x04,0x04,0x0E}},
    {'2',{0x0E,0x11,0x01,0x02,0x04,0x08,0x1F}}, {'3',{0x1F,0x02,0x04,0x02,0x01,0x11,0x0E}},
    {'4',{0x02,0x06,0x0A,0x12,0x1F,0x02,0x02}}, {'5',{0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E}},
    {'6',{0x06,0x08,0x10,0x1E,0x11,0x11,0x0E}}, {'7',{0x1F,0x01,0x02,0x04,0x08,0x08,0x08}},
    {'8',{0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E}}, {'9',{0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C}},
    {'.',{0x00,0x00,0x00,0x00,0x00,0x0C,0x0C}}, {',',{0x00,0x00,0x00,0x00,0x0C,0x04,0x08}},
    {':',{0x00,0x0C,0x0C,0x00,0x0C,0x0C,0x00}}, {';',{0x00,0x0C,0x0C,0x00,0x0C,0x04,0x08}},
    {'-',{0x00,0x00,0x00,0x1F,0x00,0x00,0x00}}, {'+',{0x00,0x04,0x04,0x1F,0x04,0x04,0x00}},
    {'/',{0x01,0x01,0x02,0x04,0x08,0x10,0x10}}, {'%',{0x18,0x19,0x02,0x04,0x08,0x13,0x03}},
    {'(',{0x02,0x04,0x08,0x08,0x08,0x04,0x02}}, {')',{0x08,0x04,0x02,0x02,0x02,0x04,0x08}},
    {'[',{0x0E,0x08,0x08,0x08,0x08,0x08,0x0E}}, {']',{0x0E,0x02,0x02,0x02,0x02,0x02,0x0E}},
    {'!',{0x04,0x04,0x04,0x04,0x04,0x00,0x04}}, {'?',{0x0E,0x11,0x01,0x02,0x04,0x00,0x04}},
    {'=',{0x00,0x00,0x1F,0x00,0x1F,0x00,0x00}}, {'_',{0x00,0x00,0x00,0x00,0x00,0x00,0x1F}},
    {'<',{0x02,0x04,0x08,0x10,0x08,0x04,0x02}}, {'>',{0x08,0x04,0x02,0x01,0x02,0x04,0x08}},
    {'|',{0x04,0x04,0x04,0x04,0x04,0x04,0x04}}, {'*',{0x00,0x04,0x15,0x0E,0x15,0x04,0x00}},
    {'#',{0x0A,0x0A,0x1F,0x0A,0x1F,0x0A,0x0A}}, {'\'',{0x04,0x04,0x08,0x00,0x00,0x00,0x00}},
    {'&',{0x0C,0x12,0x14,0x08,0x15,0x12,0x0D}}, {' ',{0,0,0,0,0,0,0}}
};
static uint8_t g_fontRows[128][7];

static void initFont() {
    std::memset(g_fontRows, 0, sizeof(g_fontRows));
    for (const Glyph& g : FONT_TABLE) std::memcpy(g_fontRows[(int)g.c], g.r, 7);
}

static float textWidth(const char* s, float scale) { return (float)std::strlen(s) * 6.0f * scale; }

// Draws text in the current 2D ortho projection. (x, y) = top-left corner.
static void drawText(float x, float y, const char* s, float scale, float r, float g, float b,
                     float a = 1.0f, bool shadow = true) {
    for (int pass = shadow ? 0 : 1; pass < 2; ++pass) {
        float ox = pass == 0 ? scale * 0.8f : 0.0f;
        float oy = pass == 0 ? scale * 0.8f : 0.0f;
        if (pass == 0) glColor4f(0.0f, 0.0f, 0.0f, a * 0.65f);
        else glColor4f(r, g, b, a);
        glBegin(GL_QUADS);
        float cx = x;
        for (const char* p = s; *p; ++p) {
            int c = (unsigned char)*p;
            if (c >= 'a' && c <= 'z') c -= 32;
            if (c > 127) c = '?';
            for (int row = 0; row < 7; ++row) {
                uint8_t bits = g_fontRows[c][row];
                int col = 0;
                while (col < 5) {
                    if (bits & (0x10 >> col)) {
                        int start = col;
                        while (col < 5 && (bits & (0x10 >> col))) ++col;
                        float x0 = cx + ox + start * scale, x1 = cx + ox + col * scale;
                        float y0 = y + oy + row * scale, y1 = y0 + scale;
                        glVertex2f(x0, y0); glVertex2f(x1, y0); glVertex2f(x1, y1); glVertex2f(x0, y1);
                    } else ++col;
                }
            }
            cx += 6.0f * scale;
        }
        glEnd();
    }
}

static void drawRect(float x0, float y0, float x1, float y1, float r, float g, float b, float a = 1.0f) {
    glColor4f(r, g, b, a);
    glBegin(GL_QUADS);
    glVertex2f(x0, y0); glVertex2f(x1, y0); glVertex2f(x1, y1); glVertex2f(x0, y1);
    glEnd();
}
static void drawRectLine(float x0, float y0, float x1, float y1, float w, float r, float g, float b, float a = 1.0f) {
    glLineWidth(w);
    glColor4f(r, g, b, a);
    glBegin(GL_LINE_LOOP);
    glVertex2f(x0, y0); glVertex2f(x1, y0); glVertex2f(x1, y1); glVertex2f(x0, y1);
    glEnd();
    glLineWidth(1.0f);
}
// Arc ribbon, angles in radians (0 = +x, increasing clockwise on screen).
static void drawArc(float cx, float cy, float r0, float r1, float a0, float a1, int segs,
                    float r, float g, float b, float a) {
    glColor4f(r, g, b, a);
    glBegin(GL_QUAD_STRIP);
    for (int i = 0; i <= segs; ++i) {
        float t = a0 + (a1 - a0) * (float)i / (float)segs;
        glVertex2f(cx + std::cos(t) * r0, cy + std::sin(t) * r0);
        glVertex2f(cx + std::cos(t) * r1, cy + std::sin(t) * r1);
    }
    glEnd();
}

static void beginOrtho() {
    glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING); glDisable(GL_FOG); glDisable(GL_TEXTURE_2D);
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity();
    glOrtho(0, g_winW, g_winH, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity();
}
static void endOrtho() {
    glMatrixMode(GL_MODELVIEW); glPopMatrix();
    glMatrixMode(GL_PROJECTION); glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glDisable(GL_BLEND);
}

// ============================================================================
//  PROCEDURAL NOISE & HD TEXTURE GENERATION
// ============================================================================
static float hashf(int x, int y, uint32_t seed) {
    uint32_t h = (uint32_t)x * 374761393u + (uint32_t)y * 668265263u + seed * 2246822519u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= h >> 16;
    return (float)(h & 0xFFFFFF) / 16777216.0f;
}
static float hash3(int x, int y, int z, uint32_t seed) {
    return hashf(x + z * 57, y + z * 131, seed);
}
// tileable 2D value noise (periods px, py in lattice cells)
static float vnoise(float x, float y, int px, int py, uint32_t seed) {
    float fx = std::floor(x), fy = std::floor(y);
    int ix = (int)fx, iy = (int)fy;
    float tx = x - fx, ty = y - fy;
    tx = tx * tx * (3 - 2 * tx);
    ty = ty * ty * (3 - 2 * ty);
    auto wx = [&](int v) { v %= px; return v < 0 ? v + px : v; };
    auto wy = [&](int v) { v %= py; return v < 0 ? v + py : v; };
    float a = hashf(wx(ix), wy(iy), seed), b = hashf(wx(ix + 1), wy(iy), seed);
    float c = hashf(wx(ix), wy(iy + 1), seed), d = hashf(wx(ix + 1), wy(iy + 1), seed);
    return lerpf(lerpf(a, b, tx), lerpf(c, d, tx), ty);
}
static float fbmTile(float u, float v, int px, int py, int oct, uint32_t seed) {
    float sum = 0, amp = 0.5f, norm = 0;
    for (int o = 0; o < oct; ++o) {
        sum += amp * vnoise(u * px, v * py, px, py, seed + o * 17u);
        norm += amp; amp *= 0.5f; px *= 2; py *= 2;
    }
    return sum / norm;
}
static float vnoise3(float x, float y, float z, uint32_t seed) {
    float fx = std::floor(x), fy = std::floor(y), fz = std::floor(z);
    int ix = (int)fx, iy = (int)fy, iz = (int)fz;
    float tx = smooth01(x - fx), ty = smooth01(y - fy), tz = smooth01(z - fz);
    float c[2][2][2];
    for (int dz = 0; dz < 2; ++dz) for (int dy = 0; dy < 2; ++dy) for (int dx = 0; dx < 2; ++dx)
        c[dz][dy][dx] = hash3(ix + dx, iy + dy, iz + dz, seed);
    float x00 = lerpf(c[0][0][0], c[0][0][1], tx), x10 = lerpf(c[0][1][0], c[0][1][1], tx);
    float x01 = lerpf(c[1][0][0], c[1][0][1], tx), x11 = lerpf(c[1][1][0], c[1][1][1], tx);
    return lerpf(lerpf(x00, x10, ty), lerpf(x01, x11, ty), tz);
}

template <class F>
static GLuint makeTexture(int w, int h, F&& fn, bool repeat = true) {
    std::vector<uint8_t> px((size_t)w * h * 4);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float c[4] = {0, 0, 0, 1};
            fn((x + 0.5f) / w, (y + 0.5f) / h, c);
            for (int k = 0; k < 4; ++k)
                px[((size_t)y * w + x) * 4 + k] = (uint8_t)(clampf(c[k], 0.0f, 1.0f) * 255.0f + 0.5f);
        }
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    GLint wrap = repeat ? GL_REPEAT : GL_CLAMP_TO_EDGE;
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    if (g_maxAniso > 1.0f) glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, g_maxAniso);
    gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGBA, w, h, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
    return tex;
}

static void initTextures() {
    g_maxAniso = 1.0f;
    glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &g_maxAniso);
    glGetError();  // swallow error if extension is missing
    if (g_maxAniso < 1.0f) g_maxAniso = 1.0f;
    g_maxAniso = std::min(g_maxAniso, 8.0f);

    // Road: high-contrast rally gravel/tarmac with micro-stones, deep wheel ruts, center crown & rally edge lines
    g_texRoad = makeTexture(512, 512, [](float u, float v, float* c) {
        float base = 0.58f + 0.34f * (fbmTile(u, v, 6, 6, 5, 11) - 0.5f);
        float fine = vnoise(u * 160, v * 160, 160, 160, 5) - 0.5f;
        float peb = vnoise(u * 80, v * 80, 80, 80, 9);
        base += fine * 0.25f + (peb > 0.78f ? 0.14f : 0.0f) - (peb < 0.14f ? 0.12f : 0.0f);
        float rutL = std::exp(-std::pow((u - 0.34f) / 0.055f, 2.0f)) + std::exp(-std::pow((u - 0.24f) / 0.045f, 2.0f));
        float rutR = std::exp(-std::pow((u - 0.66f) / 0.055f, 2.0f)) + std::exp(-std::pow((u - 0.76f) / 0.045f, 2.0f));
        float rut = clampf((rutL + rutR) * 0.65f, 0.0f, 1.0f);
        base *= 1.0f - 0.26f * rut;
        float berm = std::exp(-std::pow((u - 0.5f) / 0.08f, 2.0f));
        base *= 1.0f + 0.10f * berm + 0.06f * (1 - rut) * (fbmTile(u, v, 24, 24, 2, 3) - 0.5f);
        float edge = smooth01((std::fabs(u - 0.5f) * 2.0f - 0.78f) / 0.22f);
        base *= 1.0f - 0.28f * edge;
        c[0] = base * 1.00f; c[1] = base * 0.95f; c[2] = base * 0.88f;
        // Subtle worn white rally edge line near outer shoulders
        float lineDist = std::fabs(std::fabs(u - 0.5f) - 0.44f);
        if (lineDist < 0.012f) {
            float wear = 0.45f + 0.45f * fbmTile(u, v, 8, 16, 3, 19);
            float w = (1.0f - lineDist / 0.012f) * wear;
            c[0] = lerpf(c[0], 0.92f, w);
            c[1] = lerpf(c[1], 0.90f, w);
            c[2] = lerpf(c[2], 0.84f, w);
        }
    });
    g_texGround = makeTexture(512, 512, [](float u, float v, float* c) {
        float base = 0.68f + 0.38f * (fbmTile(u, v, 6, 6, 6, 21) - 0.5f) +
                     0.14f * (vnoise(u * 192, v * 192, 192, 192, 4) - 0.5f);
        float p1 = fbmTile(u, v, 4, 4, 3, 31), p2 = fbmTile(u, v, 5, 5, 3, 41);
        float wildflower = (vnoise(u * 96, v * 96, 96, 96, 53) > 0.88f) ? 0.08f : 0.0f;
        c[0] = base * (0.90f + 0.16f * p1) + wildflower;
        c[1] = base * (0.99f + 0.05f * p2) + wildflower * 0.8f;
        c[2] = base * (0.82f + 0.16f * p2);
    });
    g_texCurb = makeTexture(128, 128, [](float u, float v, float* c) {
        bool red = v < 0.5f;
        float grit = (hashf((int)(u * 128), (int)(v * 128), 77) - 0.5f) * 0.10f;
        float edge = 1.0f - 0.25f * smooth01((std::fabs(u - 0.5f) * 2.0f - 0.8f) / 0.2f);
        float seam = 1.0f - 0.35f * (std::exp(-std::pow((v - 0.5f) / 0.012f, 2.0f)) +
                                     std::exp(-std::pow(v / 0.012f, 2.0f)) +
                                     std::exp(-std::pow((v - 1.0f) / 0.012f, 2.0f)));
        if (red) { c[0] = 0.88f + grit; c[1] = 0.10f + grit * 0.4f; c[2] = 0.10f + grit * 0.4f; }
        else { c[0] = c[1] = c[2] = 0.95f + grit; }
        for (int i = 0; i < 3; ++i) c[i] *= edge * seam;
    });

    // Facades: one cell = 4 m wide x 3 m high (one storey) with stone quoins, cornice band, window reflection & flower box
    auto facade = [](bool wood) {
        return [wood](float u, float v, float* c) {
            float n = fbmTile(u, v, 4, 3, 4, wood ? 51 : 61);
            float wall[3];
            if (wood) {
                float plank = std::floor(v * 12.0f);
                float shade = 0.80f + 0.18f * hashf((int)plank, 3, 5);
                float grain = vnoise(u * 3, v * 60, 3, 60, 7);
                float g = shade * (0.78f + 0.30f * grain);
                float seam = std::fabs(v * 12.0f - plank - 0.5f) > 0.46f ? 0.55f : 1.0f;
                wall[0] = g * 0.78f * seam; wall[1] = g * 0.58f * seam; wall[2] = g * 0.40f * seam;
            } else {
                float g = 0.91f + 0.14f * (n - 0.5f) + 0.04f * (hashf((int)(u * 256), (int)(v * 192), 8) - 0.5f);
                wall[0] = g; wall[1] = g * 0.97f; wall[2] = g * 0.93f;
                // Subtle stone quoins along bay edges & floor cornice band at top/bottom
                if ((u < 0.045f || u > 0.955f) && std::fmod(v * 6.0f, 1.0f) < 0.78f) {
                    wall[0] *= 0.86f; wall[1] *= 0.85f; wall[2] *= 0.83f;
                }
                if (v > 0.93f) { wall[0] *= 0.84f; wall[1] *= 0.83f; wall[2] *= 0.81f; }
            }
            float dirt = 1.0f - 0.22f * (1.0f - smooth01(v / 0.11f));
            c[0] = wall[0] * dirt; c[1] = wall[1] * dirt; c[2] = wall[2] * dirt;
            const float wx0 = 0.33f, wx1 = 0.67f, wy0 = 0.30f, wy1 = 0.80f;
            const float fr = 0.025f;
            if (u > wx0 - fr && u < wx1 + fr && v > wy0 - fr && v < wy1 + fr) {
                bool inGlass = u > wx0 && u < wx1 && v > wy0 && v < wy1;
                bool mullion = std::fabs(u - 0.5f) < 0.013f || std::fabs(v - 0.55f) < 0.017f;
                if (!inGlass || mullion) { c[0] = 0.95f; c[1] = 0.94f; c[2] = 0.90f; }
                else {
                    float refl = smooth01((u - 0.34f + (v - 0.3f) * 0.85f) * 2.2f) * 0.28f;
                    float warm = smooth01(1.0f - (v - wy0) / (wy1 - wy0)) * 0.18f;
                    float g = 0.12f + 0.20f * (v - wy0) / (wy1 - wy0) + refl;
                    c[0] = g * 0.78f + warm * 1.1f; c[1] = g * 0.96f + warm * 0.75f; c[2] = g * 1.28f;
                }
            }
            // Stone window sill + Alpine flower box with red/yellow blossoms
            if (v > wy0 - 0.055f && v < wy0 - 0.012f && u > wx0 - 0.04f && u < wx1 + 0.04f) {
                c[0] = 0.36f; c[1] = 0.22f; c[2] = 0.12f;
            }
            if (v >= wy0 - 0.012f && v < wy0 + 0.045f && u > wx0 - 0.02f && u < wx1 + 0.02f) {
                float fl = hashf((int)(u * 48), (int)(v * 48), 99);
                if (fl > 0.68f)      { c[0] = 0.90f; c[1] = 0.16f; c[2] = 0.18f; }
                else if (fl > 0.52f) { c[0] = 0.95f; c[1] = 0.82f; c[2] = 0.18f; }
                else                 { c[0] = 0.18f; c[1] = 0.48f; c[2] = 0.16f; }
            }
            if (!wood) {
                bool shL = u > 0.19f && u < 0.305f, shR = u > 0.695f && u < 0.81f;
                if ((shL || shR) && v > wy0 - 0.012f && v < wy1 + 0.012f) {
                    float slat = 0.75f + 0.25f * std::sin(v * 150.0f);
                    c[0] = 0.22f * slat; c[1] = 0.42f * slat; c[2] = 0.24f * slat;
                }
            }
        };
    };
    g_texFacadeA = makeTexture(256, 192, facade(false));
    g_texFacadeB = makeTexture(256, 192, facade(true));
    g_texRoof = makeTexture(256, 256, [](float u, float v, float* c) {
        float row = v * 8.0f; float rf = std::floor(row);
        float uu = u * 8.0f + (((int)rf & 1) ? 0.5f : 0.0f);
        float uf = uu - std::floor(uu);
        float ry = row - rf;
        float g = 0.70f + 0.25f * hashf((int)std::floor(uu), (int)rf, 13) + 0.10f * (fbmTile(u, v, 8, 8, 3, 9) - 0.5f);
        float shade = 0.55f + 0.45f * ry;
        float groove = (uf < 0.04f || ry < 0.08f) ? 0.55f : 1.0f;
        float v2 = g * shade * groove;
        c[0] = v2; c[1] = v2 * 0.97f; c[2] = v2 * 0.94f;
    });
    g_texWood = makeTexture(256, 256, [](float u, float v, float* c) {
        float board = std::floor(u * 8.0f);
        float g = 0.62f + 0.22f * hashf((int)board, 2, 17) + 0.22f * (vnoise(u * 8, v * 48, 8, 48, 4) - 0.5f);
        float gap = std::fabs(u * 8.0f - board - 0.5f) > 0.47f ? 0.45f : 1.0f;
        c[0] = g * 0.80f * gap; c[1] = g * 0.60f * gap; c[2] = g * 0.42f * gap;
    });
    g_texPlaster = makeTexture(256, 256, [](float u, float v, float* c) {
        float g = 0.82f + 0.30f * (fbmTile(u, v, 6, 6, 5, 71) - 0.5f);
        c[0] = g; c[1] = g * 0.98f; c[2] = g * 0.95f;
    });
    g_texRock = makeTexture(256, 256, [](float u, float v, float* c) {
        float n = fbmTile(u, v, 5, 5, 5, 81);
        float ridge = std::fabs(fbmTile(u, v, 4, 4, 3, 91) - 0.5f);
        float g = 0.52f + 0.50f * (n - 0.5f) - 0.28f * (1.0f - smooth01(ridge * 9.0f));
        g += 0.10f * (vnoise(u * 96, v * 96, 96, 96, 2) - 0.5f);
        c[0] = g; c[1] = g * 0.98f; c[2] = g * 0.95f;
    });
    g_texBark = makeTexture(128, 128, [](float u, float v, float* c) {
        float n = vnoise(u * 8, v * 4, 8, 4, 31) * 0.6f + vnoise(u * 16, v * 8, 16, 8, 33) * 0.4f;
        float g = 0.35f + 0.65f * n;
        c[0] = g * 0.55f; c[1] = g * 0.40f; c[2] = g * 0.28f;
    });
    g_texNeedle = makeTexture(128, 128, [](float u, float v, float* c) {
        float n = vnoise(u * 24, v * 24, 24, 24, 43);
        float m = vnoise(u * 64, v * 64, 64, 64, 47);
        float g = 0.45f + 0.40f * n + 0.20f * (m - 0.5f);
        c[0] = g; c[1] = g; c[2] = g;
    });
    g_texSoft = makeTexture(64, 64, [](float u, float v, float* c) {
        float dx = u - 0.5f, dy = v - 0.5f;
        float r = std::sqrt(dx * dx + dy * dy) * 2.0f;
        float a = smooth01(1.0f - r);
        a *= 0.65f + 0.35f * vnoise(u * 6, v * 6, 6, 6, 5);
        c[0] = c[1] = c[2] = 1.0f; c[3] = a * a;
    }, false);
    g_texShadow = makeTexture(64, 64, [](float u, float v, float* c) {
        float dx = (u - 0.5f) * 2.0f, dy = (v - 0.5f) * 2.0f;
        float r = std::pow(std::fabs(dx), 2.8f) + std::pow(std::fabs(dy), 2.8f);
        c[0] = c[1] = c[2] = 0.0f; c[3] = smooth01(1.0f - r) * 0.65f;
    }, false);
    g_texCloud = makeTexture(256, 256, [](float u, float v, float* c) {
        float n = fbmTile(u, v, 4, 4, 6, 99);
        float a = smooth01((n - 0.42f) / 0.30f);
        c[0] = c[1] = c[2] = 0.80f + 0.20f * (1.0f - a); c[3] = a;
    });
    g_texTire = makeTexture(64, 64, [](float u, float v, float* c) {
        float lug = (std::fmod(u * 24.0f, 1.0f) < 0.55f) ? 1.0f : 0.55f;
        float g = (0.10f + 0.06f * vnoise(u * 32, v * 32, 32, 32, 3)) * lug;
        c[0] = g; c[1] = g; c[2] = g * 1.05f;
    });
}

// ============================================================================
//  STAGE GENERATION  (Catmull-Rom smoothing, curvature limiting, elevation,
//  pace-notes, terrain height field, road queries, collision-free scenery)
// ============================================================================
static bool g_stageClosed = false;

static inline Vec2 operator+(const Vec2& a, const Vec2& b) { return Vec2(a.x + b.x, a.y + b.y); }
static inline Vec2 operator-(const Vec2& a, const Vec2& b) { return Vec2(a.x - b.x, a.y - b.y); }
static inline Vec2 operator*(const Vec2& a, float s) { return Vec2(a.x * s, a.y * s); }
static inline float len2(const Vec2& a) { return std::sqrt(a.x * a.x + a.y * a.y); }
static inline float rand01() { return std::uniform_real_distribution<float>(0.0f, 1.0f)(g_rng); }
static inline float randRange(float a, float b) { return a + (b - a) * rand01(); }

static float terrainBase(float x, float z) {
    float h = 9.0f * std::sin(x * 0.0042f + 0.7f) * std::cos(z * 0.0039f - 0.3f) +
              3.5f * std::sin((x + z) * 0.0105f) + 2.0f * std::cos(x * 0.021f - z * 0.017f) +
              0.8f * std::sin(x * 0.06f) * std::sin(z * 0.055f);
    float edge = std::max(std::fabs(x), std::fabs(z));
    float m = smooth01((edge - 1050.0f) / 700.0f);  // mountain rim far away
    h += m * (95.0f + 55.0f * std::sin(x * 0.0031f + 1.1f) * std::sin(z * 0.0027f + 0.4f) +
              22.0f * std::sin(x * 0.011f + z * 0.009f));
    return h;
}

static Vec2 evalCatmullRom(const Vec2& p0, const Vec2& p1, const Vec2& p2, const Vec2& p3, float t) {
    float t2 = t * t, t3 = t2 * t;
    auto f = [&](float a, float b, float c, float d) {
        return 0.5f * ((2.0f * b) + (-a + c) * t + (2.0f * a - 5.0f * b + 4.0f * c - d) * t2 +
                       (-a + 3.0f * b - 3.0f * c + d) * t3);
    };
    return Vec2(f(p0.x, p1.x, p2.x, p3.x), f(p0.y, p1.y, p2.y, p3.y));
}

static std::vector<Vec2> resampleUniform(const std::vector<Vec2>& in, bool closed, float step) {
    size_t n = in.size();
    if (n < 2) return in;
    size_t segs = closed ? n : n - 1;
    std::vector<float> cum(segs + 1, 0.0f);
    for (size_t i = 0; i < segs; ++i) cum[i + 1] = cum[i] + len2(in[(i + 1) % n] - in[i]);
    float total = cum[segs];
    if (total < 1e-3f) return in;
    int count;
    float spacing;
    if (closed) { count = std::max(3, (int)std::lround(total / step)); spacing = total / count; }
    else { count = std::max(2, (int)std::floor(total / step) + 1); spacing = step; }
    std::vector<Vec2> out;
    out.reserve(count + 1);
    size_t seg = 0;
    for (int i = 0; i < count; ++i) {
        float s = i * spacing;
        while (seg + 1 < segs && cum[seg + 1] < s) ++seg;
        float len = cum[seg + 1] - cum[seg];
        float t = len > 1e-6f ? clampf((s - cum[seg]) / len, 0.0f, 1.0f) : 0.0f;
        Vec2 a = in[seg], b = in[(seg + 1) % n];
        out.push_back(a + (b - a) * t);
    }
    if (!closed) {
        float lastS = (count - 1) * spacing;
        if (total - lastS > step * 0.35f) out.push_back(in.back());
        else out.back() = in.back();
    }
    return out;
}

// Iterative Laplacian relaxation: guarantees a minimum turning radius so the extruded
// road ribbon can never fold over itself at a sharp user-drawn corner.
static void relaxCurvature(std::vector<Vec2>& P, bool closed, float step, float minRadius) {
    int n = (int)P.size();
    if (n < 5) return;
    float maxAng = step / minRadius;
    std::vector<Vec2> Q;
    for (int it = 0; it < 4000; ++it) {
        bool any = false;
        Q = P;
        int i0 = closed ? 0 : 1, i1 = closed ? n : n - 1;
        for (int i = i0; i < i1; ++i) {
            Vec2 prev = P[(i - 1 + n) % n], next = P[(i + 1) % n];
            Vec2 a = P[i] - prev, b = next - P[i];
            float ang = std::fabs(std::atan2(a.x * b.y - a.y * b.x, a.x * b.x + a.y * b.y));
            if (ang > maxAng) {
                Vec2 mid = (prev + next) * 0.5f;
                Q[i] = P[i] + (mid - P[i]) * 0.5f;
                any = true;
            }
        }
        P.swap(Q);
        if (!any) break;
    }
}

static std::vector<Vec2> buildCenterline(const std::vector<Vec2>& rawIn, bool closed) {
    std::vector<Vec2> pts;
    for (const Vec2& p : rawIn)
        if (pts.empty() || len2(p - pts.back()) > 0.5f) pts.push_back(p);
    if (closed && pts.size() > 2 && len2(pts.front() - pts.back()) < 1.0f) pts.pop_back();
    if (pts.size() < 2) return {};

    // 1. Control polygon at 18m spacing with light jitter removal so spline passes through user path
    std::vector<Vec2> ctrl = resampleUniform(pts, closed, 18.0f);
    int n = (int)ctrl.size();
    if (n >= 4) {
        std::vector<Vec2> tmp = ctrl;
        for (int i = 0; i < n; ++i) {
            if (!closed && (i == 0 || i == n - 1)) continue;
            tmp[i] = ctrl[i] * 0.72f + (ctrl[(i + n - 1) % n] + ctrl[(i + 1) % n]) * 0.14f;
        }
        ctrl.swap(tmp);
    }
    if (n < 3) return resampleUniform(ctrl, closed, STAGE_STEP);

    // 2. Catmull-Rom spline through the control points
    std::vector<Vec2> dense;
    int segs = closed ? n : n - 1;
    auto at = [&](int i) -> Vec2 {
        if (closed) return ctrl[((i % n) + n) % n];
        return ctrl[std::max(0, std::min(n - 1, i))];
    };
    for (int i = 0; i < segs; ++i)
        for (int s = 0; s < 12; ++s)
            dense.push_back(evalCatmullRom(at(i - 1), at(i), at(i + 1), at(i + 2), s / 12.0f));
    if (!closed) dense.push_back(ctrl.back());

    // 3. Uniform arc-length resample, minimum radius 15m (prevents curb self-overlap), resample again
    std::vector<Vec2> line = resampleUniform(dense, closed, STAGE_STEP);
    relaxCurvature(line, closed, STAGE_STEP, 15.0f);
    return resampleUniform(line, closed, STAGE_STEP);
}

// ---------------------------------------------------------------- grids ----
static inline int gridCell(float v, float cell, int n) {
    int c = (int)std::floor((v + TERRAIN_HALF) / cell);
    return std::max(0, std::min(n - 1, c));
}

static void buildNodeGrid() {
    g_nodeGrid.assign((size_t)NODE_GRID_N * NODE_GRID_N, std::vector<int>());
    for (size_t i = 0; i < g_nodes.size(); ++i) {
        int cx = gridCell(g_nodes[i].pos.x, NODE_CELL, NODE_GRID_N);
        int cz = gridCell(g_nodes[i].pos.z, NODE_CELL, NODE_GRID_N);
        g_nodeGrid[(size_t)cz * NODE_GRID_N + cx].push_back((int)i);
    }
}

// Distance (XZ) from a point to the road centre line, with interpolated road height.
static bool nearestRoadInfo(float x, float z, float maxDist, float* outDist, float* outY, int* outIdx) {
    int N = (int)g_nodes.size();
    if (N < 2) return false;
    int cx = gridCell(x, NODE_CELL, NODE_GRID_N), cz = gridCell(z, NODE_CELL, NODE_GRID_N);
    int rc = (int)std::ceil(maxDist / NODE_CELL);
    float bestD2 = maxDist * maxDist;
    int best = -1;
    for (int gz = std::max(0, cz - rc); gz <= std::min(NODE_GRID_N - 1, cz + rc); ++gz)
        for (int gx = std::max(0, cx - rc); gx <= std::min(NODE_GRID_N - 1, cx + rc); ++gx)
            for (int i : g_nodeGrid[(size_t)gz * NODE_GRID_N + gx]) {
                float dx = x - g_nodes[i].pos.x, dz = z - g_nodes[i].pos.z;
                float d2 = dx * dx + dz * dz;
                if (d2 < bestD2) { bestD2 = d2; best = i; }
            }
    if (best < 0) return false;
    float bd2 = bestD2, by = g_nodes[best].pos.y;
    for (int s = -1; s <= 0; ++s) {
        int a = best + s, b = best + s + 1;
        if (g_stageClosed) { a = (a + N) % N; b = (b + N) % N; }
        else if (a < 0 || b >= N) continue;
        const Vec3& pa = g_nodes[a].pos;
        const Vec3& pb = g_nodes[b].pos;
        float abx = pb.x - pa.x, abz = pb.z - pa.z;
        float l2 = abx * abx + abz * abz;
        float t = l2 > 1e-6f ? clampf(((x - pa.x) * abx + (z - pa.z) * abz) / l2, 0.0f, 1.0f) : 0.0f;
        float qx = pa.x + abx * t - x, qz = pa.z + abz * t - z;
        float d2 = qx * qx + qz * qz;
        if (d2 < bd2) { bd2 = d2; by = lerpf(pa.y, pb.y, t); }
    }
    if (outDist) *outDist = std::sqrt(bd2);
    if (outY) *outY = by;
    if (outIdx) *outIdx = best;
    return true;
}

static void buildHeightfield() {
    g_hf.assign((size_t)HF_N * HF_N, 0.0f);
    g_hfRoadDist.assign((size_t)HF_N * HF_N, 99.0f);
    for (int iz = 0; iz < HF_N; ++iz)
        for (int ix = 0; ix < HF_N; ++ix) {
            float x = -TERRAIN_HALF + ix * TERRAIN_STEP, z = -TERRAIN_HALF + iz * TERRAIN_STEP;
            float base = terrainBase(x, z), d = 99.0f, ry = 0.0f;
            float h = base;
            if (nearestRoadInfo(x, z, 85.0f, &d, &ry, nullptr)) {
                float w = smooth01((d - 22.0f) / 55.0f);   // deep bed under road+curbs so terrain never pokes through
                h = lerpf(ry - 1.75f, base, w);
            } else d = 99.0f;
            g_hf[(size_t)iz * HF_N + ix] = h;
            g_hfRoadDist[(size_t)iz * HF_N + ix] = d;
        }
}

static float terrainHeight(float x, float z) {
    if (g_hf.empty()) return terrainBase(x, z);
    float fx = clampf((x + TERRAIN_HALF) / TERRAIN_STEP, 0.0f, HF_N - 1.001f);
    float fz = clampf((z + TERRAIN_HALF) / TERRAIN_STEP, 0.0f, HF_N - 1.001f);
    int ix = (int)fx, iz = (int)fz;
    float tx = fx - ix, tz = fz - iz;
    const float* h = g_hf.data();
    float a = h[(size_t)iz * HF_N + ix], b = h[(size_t)iz * HF_N + ix + 1];
    float c = h[(size_t)(iz + 1) * HF_N + ix], d = h[(size_t)(iz + 1) * HF_N + ix + 1];
    return lerpf(lerpf(a, b, tx), lerpf(c, d, tx), tz);
}
static float terrainRoadDist(float x, float z) {
    if (g_hfRoadDist.empty()) return 99.0f;
    int ix = (int)clampf(std::floor((x + TERRAIN_HALF) / TERRAIN_STEP + 0.5f), 0.0f, HF_N - 1.0f);
    int iz = (int)clampf(std::floor((z + TERRAIN_HALF) / TERRAIN_STEP + 0.5f), 0.0f, HF_N - 1.0f);
    return g_hfRoadDist[(size_t)iz * HF_N + ix];
}

// Full road query: closest segment (windowed around `hint` for speed and to avoid
// jumping to a neighbouring leg of the track), lateral offset, C1-smooth height.
static bool queryRoad(float x, float z, int hint, RoadQuery& q) {
    int N = (int)g_nodes.size();
    if (N < 2) return false;
    int segs = g_stageClosed ? N : N - 1;
    float bestD2 = 1e30f;
    int bestI = -1;
    float bestT = 0;
    auto scan = [&](int from, int count) {
        for (int k = 0; k < count; ++k) {
            int i = from + k;
            if (g_stageClosed) i = ((i % segs) + segs) % segs;
            else if (i < 0 || i >= segs) continue;
            const Vec3& pa = g_nodes[i].pos;
            const Vec3& pb = g_nodes[(i + 1) % N].pos;
            float abx = pb.x - pa.x, abz = pb.z - pa.z;
            float l2 = abx * abx + abz * abz;
            float t = l2 > 1e-6f ? clampf(((x - pa.x) * abx + (z - pa.z) * abz) / l2, 0.0f, 1.0f) : 0.0f;
            float qx = pa.x + abx * t - x, qz = pa.z + abz * t - z;
            float d2 = qx * qx + qz * qz;
            if (d2 < bestD2) { bestD2 = d2; bestI = i; bestT = t; }
        }
    };
    if (hint >= 0) scan(hint - 60, 121);
    if (bestI < 0 || bestD2 > 45.0f * 45.0f) { bestD2 = 1e30f; bestI = -1; scan(0, segs); }
    if (bestI < 0) return false;

    const StageNode& A = g_nodes[bestI];
    const StageNode& B = g_nodes[(bestI + 1) % N];
    float t = bestT;
    float dx = B.pos.x - A.pos.x, dz = B.pos.z - A.pos.z;
    float L = std::sqrt(dx * dx + dz * dz);
    float t2 = t * t, t3 = t2 * t;
    float m0 = A.slope * L, m1 = B.slope * L;
    q.roadY = (2 * t3 - 3 * t2 + 1) * A.pos.y + (t3 - 2 * t2 + t) * m0 + (-2 * t3 + 3 * t2) * B.pos.y + (t3 - t2) * m1;
    q.idx = bestI; q.t = t; q.distSq = bestD2;
    q.closest = Vec3(A.pos.x + dx * t, q.roadY, A.pos.z + dz * t);
    q.right = (A.right * (1.0f - t) + B.right * t).normalized();
    q.fwd = (A.forward * (1.0f - t) + B.forward * t).normalized();
    q.lateral = (x - q.closest.x) * q.right.x + (z - q.closest.z) * q.right.z;
    q.halfW = 0.5f * lerpf(A.roadWidth, B.roadWidth, t);
    return true;
}

// Ground height anywhere: road surface -> sloped bank -> terrain. Consistent with the meshes.
static float worldHeightAt(float x, float z, int hint, RoadQuery* qOut = nullptr) {
    RoadQuery q;
    if (queryRoad(x, z, hint, q)) {
        if (qOut) *qOut = q;
        float ex = std::fabs(q.lateral) - (q.halfW + CURB_W);
        if (ex <= 0.0f) return q.roadY;
        if (ex < SHOULDER_W) {
            float s = q.lateral > 0 ? 1.0f : -1.0f;
            Vec3 outer = q.closest + q.right * (s * (q.halfW + CURB_W + SHOULDER_W));
            return lerpf(q.roadY, terrainHeight(outer.x, outer.z), ex / SHOULDER_W);
        }
    }
    return terrainHeight(x, z);
}

// -------------------------------------------------------- stage building ----
static void computeFrames() {
    int N = (int)g_nodes.size();
    for (int i = 0; i < N; ++i) {
        int a = g_stageClosed ? (i - 1 + N) % N : std::max(0, i - 1);
        int b = g_stageClosed ? (i + 1) % N : std::min(N - 1, i + 1);
        Vec3 f = (g_nodes[b].pos - g_nodes[a].pos).normalized();
        Vec3 right = cross(f, Vec3(0, 1, 0)).normalized();   // visual right when looking along f
        Vec3 up = cross(right, f).normalized();
        g_nodes[i].forward = f;
        g_nodes[i].right = right;
        g_nodes[i].up = up;
        g_nodes[i].slope = f.y / std::max(0.01f, f.lengthXZ());
    }
}

static void buildPaceNotes() {
    g_events.clear();
    int N = (int)g_nodes.size();
    int i = 0;
    while (i < N) {
        float c = g_nodes[i].curvature;
        if (std::fabs(c) > 0.0125f) {
            int dir = c > 0 ? 1 : -1, s = i, e = i;
            float maxc = 0, angle = 0;
            while (i < N && std::fabs(g_nodes[i].curvature) > 0.009f && (g_nodes[i].curvature > 0) == (dir > 0)) {
                maxc = std::max(maxc, std::fabs(g_nodes[i].curvature));
                angle += std::fabs(g_nodes[i].curvature) * STAGE_STEP;
                e = i; ++i;
            }
            if (angle > 0.35f) {
                int sev = maxc > 0.036f ? 1 : (maxc > 0.022f ? 2 : 3);
                g_events.push_back({EVT_CORNER, dir, sev, s, e});
            }
        } else ++i;
    }
}

static bool buildStage(const std::vector<Vec2>& worldPts, bool closed) {
    std::vector<Vec2> line = buildCenterline(worldPts, closed);
    int N = (int)line.size();
    if (N < 24) return false;
    g_stageClosed = closed;
    g_nodes.assign(N, StageNode());
    for (int i = 0; i < N; ++i) {
        g_nodes[i].pos = Vec3(line[i].x, terrainBase(line[i].x, line[i].y), line[i].y);
        g_nodes[i].dist = i == 0 ? 0.0f : g_nodes[i - 1].dist + (g_nodes[i].pos - g_nodes[i - 1].pos).lengthXZ();
    }
    g_stageLength = g_nodes.back().dist + (closed ? (g_nodes[0].pos - g_nodes.back().pos).lengthXZ() : 0.0f);

    // signed horizontal curvature (+ = left)
    const int K = 3;
    for (int i = 0; i < N; ++i) {
        int a = closed ? (i - K + N * 4) % N : std::max(0, i - K);
        int b = closed ? (i + K) % N : std::min(N - 1, i + K);
        Vec3 f1 = Vec3(g_nodes[i].pos.x - g_nodes[a].pos.x, 0, g_nodes[i].pos.z - g_nodes[a].pos.z);
        Vec3 f2 = Vec3(g_nodes[b].pos.x - g_nodes[i].pos.x, 0, g_nodes[b].pos.z - g_nodes[i].pos.z);
        float l1 = f1.length(), l2 = f2.length();
        if (l1 < 1e-3f || l2 < 1e-3f) { g_nodes[i].curvature = 0; continue; }
        f1 = f1 * (1.0f / l1); f2 = f2 * (1.0f / l2);
        float ang = std::atan2(f1.z * f2.x - f1.x * f2.z, f1.x * f2.x + f1.z * f2.z);
        g_nodes[i].curvature = ang / (0.5f * (l1 + l2));
        g_nodes[i].roadWidth = 9.0f;
    }
    for (int i = 0; i < N; ++i)  // wider road in tight corners
        g_nodes[i].roadWidth = 9.0f + 1.8f * smooth01(std::fabs(g_nodes[i].curvature) / 0.04f);

    // jump crests on long straights
    std::vector<int> crests;
    int target = (int)clampf(g_stageLength / 520.0f, 1.0f, 7.0f);
    float lastS = -1000.0f;
    for (int i = 30; i < N - 30 && (int)crests.size() < target; ++i) {
        if (g_nodes[i].dist - lastS < 280.0f) continue;
        bool straight = true;
        for (int k = -22; k <= 22 && straight; k += 2) {
            int j = closed ? (i + k + N) % N : i + k;
            if (std::fabs(g_nodes[j].curvature) > 0.0045f) straight = false;
        }
        if (straight && rand01() < 0.10f) { crests.push_back(i); lastS = g_nodes[i].dist; }
    }
    g_events.clear();
    for (int ci : crests) {
        float H = randRange(1.5f, 2.3f), w = 13.0f;
        for (int j = 0; j < N; ++j) {
            float d = std::fabs(g_nodes[j].dist - g_nodes[ci].dist);
            if (closed) d = std::min(d, g_stageLength - d);
            if (d < 4.0f * w) g_nodes[j].pos.y += H * std::exp(-(d / w) * (d / w));
        }
    }
    computeFrames();
    buildPaceNotes();
    for (int ci : crests) g_events.push_back({EVT_JUMP, 0, 4, std::max(0, ci - 5), std::min(N - 1, ci + 2)});
    std::sort(g_events.begin(), g_events.end(), [](const StageEvent& a, const StageEvent& b) { return a.startIdx < b.startIdx; });

    buildNodeGrid();
    buildHeightfield();
    return true;
}

// ----------------------------------------------------- climate & scenery ----
static void selectClimate(bool forceDifferent) {
    int next = std::uniform_int_distribution<int>(0, 3)(g_rng);
    if (forceDifferent && next == g_climate) next = (next + 1 + (int)(rand01() * 2.99f)) % 4;
    g_climate = next;
    g_surfaceGrip = CLIMATES[g_climate].surfaceGrip;
}

static float forestDensity(float x, float z) {
    float n = fbmTile((x + TERRAIN_HALF) / (2 * TERRAIN_HALF), (z + TERRAIN_HALF) / (2 * TERRAIN_HALF), 9, 9, 3, 123);
    return 0.12f + 0.88f * smooth01((n - 0.38f) / 0.28f);
}

static void buildSolidGrid() {
    g_solidGrid.assign((size_t)SOLID_GRID_N * SOLID_GRID_N, std::vector<int>());
    for (size_t i = 0; i < g_scenery.size(); ++i) {
        const SceneryObject& o = g_scenery[i];
        if (!o.solid) continue;
        float r = o.solid == 2 ? 0.5f * std::sqrt(o.w * o.w + o.d * o.d) : o.radius;
        int x0 = gridCell(o.pos.x - r, SOLID_CELL, SOLID_GRID_N), x1 = gridCell(o.pos.x + r, SOLID_CELL, SOLID_GRID_N);
        int z0 = gridCell(o.pos.z - r, SOLID_CELL, SOLID_GRID_N), z1 = gridCell(o.pos.z + r, SOLID_CELL, SOLID_GRID_N);
        for (int gz = z0; gz <= z1; ++gz)
            for (int gx = x0; gx <= x1; ++gx) g_solidGrid[(size_t)gz * SOLID_GRID_N + gx].push_back((int)i);
    }
}

static void populateScenery() {
    g_scenery.clear();
    int N = (int)g_nodes.size();
    if (N < 24) return;
    const ClimateProfile& cp = CLIMATES[g_climate];

    constexpr float PCELL = 16.0f;
    constexpr int PN = (int)(2 * TERRAIN_HALF / PCELL) + 1;
    std::vector<std::vector<int>> pgrid((size_t)PN * PN);

    auto roadDist = [&](float x, float z) -> float {
        float d = 1e9f;
        if (nearestRoadInfo(x, z, 90.0f, &d, nullptr, nullptr)) return d;
        return 1e9f;
    };
    // collision-free test against the road and every earlier object (minimum distance algorithm)
    auto isFree = [&](const SceneryObject& o, float minRoad, float gap) -> bool {
        if (std::fabs(o.pos.x) > 1300.0f || std::fabs(o.pos.z) > 1300.0f) return false;
        if (roadDist(o.pos.x, o.pos.z) < minRoad) return false;
        int cx = gridCell(o.pos.x, PCELL, PN), cz = gridCell(o.pos.z, PCELL, PN);
        int rc = (int)std::ceil((o.radius + 14.0f + gap) / PCELL);
        for (int gz = std::max(0, cz - rc); gz <= std::min(PN - 1, cz + rc); ++gz)
            for (int gx = std::max(0, cx - rc); gx <= std::min(PN - 1, cx + rc); ++gx)
                for (int idx : pgrid[(size_t)gz * PN + gx]) {
                    const SceneryObject& e = g_scenery[idx];
                    float dx = o.pos.x - e.pos.x, dz = o.pos.z - e.pos.z;
                    float req = o.radius + e.radius + gap;
                    if (dx * dx + dz * dz < req * req) return false;
                }
        return true;
    };
    auto commit = [&](SceneryObject& o, int hintIdx) {
        o.pos.y = worldHeightAt(o.pos.x, o.pos.z, hintIdx);
        g_scenery.push_back(o);
        int cx = gridCell(o.pos.x, PCELL, PN), cz = gridCell(o.pos.z, PCELL, PN);
        pgrid[(size_t)cz * PN + cx].push_back((int)g_scenery.size() - 1);
    };
    auto faceRoad = [&](SceneryObject& o, int idx) {
        Vec3 d = g_nodes[idx].pos - o.pos;
        o.yaw = std::atan2(d.x, d.z) * RAD2DEG;
    };

    // --- start / finish arches
    {
        SceneryObject a; a.kind = SCN_ARCH; a.archType = 0; a.radius = 6.0f;
        int idx = std::min(10, N - 1);
        a.pos = g_nodes[idx].pos; a.yaw = std::atan2(g_nodes[idx].forward.x, g_nodes[idx].forward.z) * RAD2DEG;
        a.w = g_nodes[idx].roadWidth + 3.0f; a.h = 7.0f; a.d = 1.0f;
        g_scenery.push_back(a);
        if (!g_stageClosed) {
            SceneryObject b = a; b.archType = 1; idx = N - 8;
            b.pos = g_nodes[idx].pos; b.yaw = std::atan2(g_nodes[idx].forward.x, g_nodes[idx].forward.z) * RAD2DEG;
            b.w = g_nodes[idx].roadWidth + 3.0f;
            g_scenery.push_back(b);
        }
    }

    // --- grandstands & hay bales at corners, grandstands at jumps
    for (const StageEvent& e : g_events) {
        int mid = (e.startIdx + e.endIdx) / 2;
        const StageNode& nd = g_nodes[mid];
        if (e.kind == EVT_CORNER) {
            float outer = e.dir > 0 ? 1.0f : -1.0f;   // outside of the turn
            if (e.sev <= 2 && rand01() < 0.80f) {
                SceneryObject g; g.kind = SCN_GRANDSTAND; g.w = 16.0f; g.d = 7.5f; g.h = 7.0f;
                g.radius = 9.5f; g.solid = 2; g.variant = (int)(rand01() * 3.0f);
                float off = nd.roadWidth * 0.5f + CURB_W + SHOULDER_W + g.d * 0.5f + 1.5f;
                g.pos = nd.pos + nd.right * (outer * off);
                faceRoad(g, mid);
                if (isFree(g, 19.5f, 2.0f)) commit(g, mid);
            }
            if (e.sev <= 3)
                for (int i = e.startIdx; i <= e.endIdx; i += 3) {
                    const StageNode& n2 = g_nodes[i];
                    SceneryObject h; h.kind = SCN_HAYBALE; h.w = 0.9f; h.d = 1.4f; h.h = 0.9f; h.radius = 0.9f;
                    h.solid = 1; h.variant = (int)(rand01() * 3.0f);
                    float off = n2.roadWidth * 0.5f + CURB_W + 1.6f;
                    h.pos = n2.pos + n2.right * (outer * off);
                    h.yaw = std::atan2(n2.forward.x, n2.forward.z) * RAD2DEG;
                    if (isFree(h, 5.6f, 0.15f)) commit(h, i);
                }
            if (e.sev <= 2) {
                int step = std::max(2, (e.endIdx - e.startIdx) / 4);
                for (int i = e.startIdx + 1; i <= e.endIdx - 1; i += step) {
                    const StageNode& n2 = g_nodes[i];
                    SceneryObject ch; ch.kind = SCN_CHEVRON;
                    ch.variant = (e.dir > 0) ? -1 : 1; // pointing left or right
                    ch.solid = 1; ch.radius = 0.5f;
                    float off = n2.roadWidth * 0.5f + CURB_W + SHOULDER_W + 0.95f;
                    ch.pos = n2.pos + n2.right * (outer * off);
                    ch.yaw = std::atan2(n2.forward.x, n2.forward.z) * RAD2DEG;
                    if (isFree(ch, 4.8f, 0.15f)) commit(ch, i);
                }
            }
        } else {
            for (int side = -1; side <= 1; side += 2) {
                if (rand01() < 0.65f) {
                    SceneryObject g; g.kind = SCN_GRANDSTAND; g.w = 16.0f; g.d = 7.5f; g.h = 7.0f;
                    g.radius = 9.5f; g.solid = 2; g.variant = (int)(rand01() * 3.0f);
                    float off = nd.roadWidth * 0.5f + CURB_W + SHOULDER_W + g.d * 0.5f + 2.0f;
                    g.pos = nd.pos + nd.right * (side * off) + nd.forward * randRange(-8.0f, 8.0f);
                    faceRoad(g, mid);
                    if (isFree(g, 19.5f, 2.0f)) commit(g, mid);
                }
            }
        }
    }

    // --- HD buildings (houses, chalets, barns, hotels) away from the road
    static const float palette[6][3] = {{0.96f, 0.93f, 0.86f}, {0.96f, 0.80f, 0.66f}, {0.92f, 0.88f, 0.60f},
                                        {0.78f, 0.84f, 0.90f}, {0.90f, 0.68f, 0.60f}, {0.86f, 0.88f, 0.84f}};
    for (int i = 10; i < N - 10; i += 8)
        for (int side = -1; side <= 1; side += 2) {
            if (rand01() > 0.17f) continue;
            for (int attempt = 0; attempt < 3; ++attempt) {
                SceneryObject b; b.kind = SCN_BUILDING;
                b.variant = (int)(rand01() * 4.0f);  // 0 house, 1 wood chalet, 2 barn, 3 hotel
                float wv[3] = {8.0f, 12.0f, 16.0f};
                b.w = wv[(int)(rand01() * 2.99f)];
                b.d = rand01() < 0.5f ? 8.0f : 12.0f;
                int floors = b.variant == 3 ? 3 : (b.variant == 2 ? 1 : 2);
                if (b.variant == 0 && rand01() < 0.3f) floors = 3;
                b.h = floors * 3.0f;
                b.radius = 0.5f * std::sqrt(b.w * b.w + b.d * b.d) + 1.5f;
                b.solid = 2;
                const float* pc = palette[(int)(rand01() * 5.99f)];
                if (b.variant == 1) { b.color[0] = 0.95f; b.color[1] = 0.85f; b.color[2] = 0.75f; }
                else if (b.variant == 2) { b.color[0] = 0.85f; b.color[1] = 0.62f; b.color[2] = 0.50f; }
                else { b.color[0] = pc[0]; b.color[1] = pc[1]; b.color[2] = pc[2]; }
                float off = randRange(22.0f, 55.0f);
                b.pos = g_nodes[i].pos + g_nodes[i].right * (side * off) + g_nodes[i].forward * randRange(-12.0f, 12.0f);
                faceRoad(b, i);
                if (isFree(b, 14.0f + b.radius * 0.7f, 3.0f)) { commit(b, i); break; }
            }
        }

    // --- rocks
    int maxRocks = 700;
    for (int i = 0; i < N && maxRocks > 0; i += 3)
        for (int side = -1; side <= 1; side += 2) {
            if (rand01() > 0.20f) continue;
            SceneryObject r; r.kind = SCN_ROCK; r.variant = (int)(rand01() * 2.99f);
            r.scale = rand01() < 0.15f ? randRange(2.0f, 3.4f) : randRange(0.7f, 1.8f);
            r.radius = 1.5f * r.scale; r.solid = 1;
            r.yaw = rand01() * 360.0f;
            float off = 7.0f + std::pow(rand01(), 1.5f) * 40.0f;
            r.pos = g_nodes[i].pos + g_nodes[i].right * (side * off);
            r.color[0] = r.color[1] = r.color[2] = randRange(0.75f, 1.05f);
            if (isFree(r, 6.8f + r.radius, 1.0f)) { commit(r, i); --maxRocks; }
        }

    // --- trees (clustered by a forest density field) and bushes
    int maxTrees = 7000;
    float coniferShare = cp.snow ? 0.85f : (g_climate == CLIMATE_HEAVY_RAIN_MUD ? 0.6f : 0.55f);
    for (int i = 0; i < N && maxTrees > 0; i += 2)
        for (int side = -1; side <= 1; side += 2)
            for (int attempt = 0; attempt < 4; ++attempt) {
                float off = 7.6f + std::pow(rand01(), 1.7f) * 85.0f;
                Vec3 p = g_nodes[i].pos + g_nodes[i].right * (side * off) + g_nodes[i].forward * randRange(-3.5f, 3.5f);
                float dens = forestDensity(p.x, p.z);
                if (rand01() > dens * (off < 30.0f ? 0.85f : 0.65f)) continue;
                SceneryObject t;
                t.pos = p;
                float roll = rand01();
                if (roll < 0.10f && off < 40.0f) {
                    t.kind = SCN_BUSH; t.scale = randRange(0.7f, 1.4f); t.radius = 0.9f * t.scale; t.solid = 0;
                } else if (rand01() < coniferShare) {
                    t.kind = SCN_CONIFER; t.scale = randRange(0.8f, 1.7f); t.radius = 1.7f * t.scale; t.solid = 1;
                } else {
                    t.kind = SCN_BROADLEAF; t.scale = randRange(0.8f, 1.5f); t.radius = 2.2f * t.scale; t.solid = 1;
                }
                t.yaw = rand01() * 360.0f;
                t.variant = (int)(rand01() * 3.0f);
                float v = randRange(0.80f, 1.15f);
                float autumn = (!cp.snow && t.kind == SCN_BROADLEAF && rand01() < 0.18f) ? 1.0f : 0.0f;
                t.color[0] = cp.foliage[0] * v + autumn * 0.55f;
                t.color[1] = cp.foliage[1] * v + autumn * 0.12f;
                t.color[2] = cp.foliage[2] * v * (1.0f - 0.6f * autumn);
                float clear = t.kind == SCN_BUSH ? 6.9f : 7.4f;
                if (isFree(t, clear + (t.solid ? 0.45f * t.scale : 0.0f), t.kind == SCN_BUSH ? 0.2f : 0.6f)) {
                    commit(t, i); --maxTrees;
                }
            }
    buildSolidGrid();
}

// canvas pixel -> world metre transform (uniform scale, drawing centred, fills +-780 m)
static void canvasTransform(const std::vector<Vec2>& pts, Vec2& centre, float& metresPerPx) {
    float minX = 1e9f, maxX = -1e9f, minY = 1e9f, maxY = -1e9f;
    for (const Vec2& p : pts) { minX = std::min(minX, p.x); maxX = std::max(maxX, p.x); minY = std::min(minY, p.y); maxY = std::max(maxY, p.y); }
    float span = std::max(120.0f, std::max(maxX - minX, maxY - minY));
    centre = Vec2(0.5f * (minX + maxX), 0.5f * (minY + maxY));
    metresPerPx = 1560.0f / span;
}

static std::vector<Vec2> defaultCanvasPoints() {
    std::vector<Vec2> pts;
    float cx = g_winW * 0.5f, cy = g_winH * 0.5f, rx = g_winW * 0.34f, ry = g_winH * 0.30f;
    const int n = 28;
    for (int i = 0; i < n; ++i) {
        float a = 2.0f * PI * i / n;
        float wob = 1.0f + 0.22f * std::sin(a * 3.0f + 0.5f) + 0.10f * std::cos(a * 5.0f);
        pts.push_back(Vec2(cx + std::cos(a) * rx * wob, cy + std::sin(a) * ry * wob));
    }
    return pts;
}

// Build the whole stage from the 2D canvas drawing (pure data, no GL).
static bool generateStageFromCanvas(const std::vector<Vec2>& canvas, bool closed) {
    Vec2 c; float mpp;
    canvasTransform(canvas, c, mpp);
    std::vector<Vec2> world;
    for (const Vec2& p : canvas) {
        Vec2 w((p.x - c.x) * mpp, (p.y - c.y) * mpp);
        w.x = clampf(w.x, -900.0f, 900.0f); w.y = clampf(w.y, -900.0f, 900.0f);
        world.push_back(w);
    }
    if (!buildStage(world, closed)) return false;
    selectClimate(false);
    populateScenery();
    return true;
}

// ============================================================================
//  GL WORLD BUILDING: sky dome, terrain, road mesh extrusion, scenery models
// ============================================================================
static inline void glV(const Vec3& v) { glVertex3f(v.x, v.y, v.z); }
static inline void glN(const Vec3& v) { glNormal3f(v.x, v.y, v.z); }
static inline void texOn(GLuint t) { glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, t); }
static inline void texOff() { glDisable(GL_TEXTURE_2D); }
static void mixColor(const float* a, const float* b, float t, float* out) {
    for (int i = 0; i < 3; ++i) out[i] = lerpf(a[i], b[i], t);
}

static void quadTex(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d, const Vec3& n,
                    float u0, float v0, float u1, float v1) {
    glN(n);
    glTexCoord2f(u0, v0); glV(a);
    glTexCoord2f(u1, v0); glV(b);
    glTexCoord2f(u1, v1); glV(c);
    glTexCoord2f(u0, v1); glV(d);
}

// Axis-aligned box with normals; if tex scales are > 0 texture coordinates are metric.
static void drawBox(float x0, float y0, float z0, float x1, float y1, float z1, float us = 0.0f, float vs = 0.0f) {
    float dx = x1 - x0, dy = y1 - y0, dz = z1 - z0;
    glBegin(GL_QUADS);
    // +Z
    quadTex(Vec3(x0, y0, z1), Vec3(x1, y0, z1), Vec3(x1, y1, z1), Vec3(x0, y1, z1), Vec3(0, 0, 1), 0, 0, dx * us, dy * vs);
    // -Z
    quadTex(Vec3(x1, y0, z0), Vec3(x0, y0, z0), Vec3(x0, y1, z0), Vec3(x1, y1, z0), Vec3(0, 0, -1), 0, 0, dx * us, dy * vs);
    // +X
    quadTex(Vec3(x1, y0, z1), Vec3(x1, y0, z0), Vec3(x1, y1, z0), Vec3(x1, y1, z1), Vec3(1, 0, 0), 0, 0, dz * us, dy * vs);
    // -X
    quadTex(Vec3(x0, y0, z0), Vec3(x0, y0, z1), Vec3(x0, y1, z1), Vec3(x0, y1, z0), Vec3(-1, 0, 0), 0, 0, dz * us, dy * vs);
    // +Y
    quadTex(Vec3(x0, y1, z1), Vec3(x1, y1, z1), Vec3(x1, y1, z0), Vec3(x0, y1, z0), Vec3(0, 1, 0), 0, 0, dx * us, dz * vs);
    // -Y
    quadTex(Vec3(x0, y0, z0), Vec3(x1, y0, z0), Vec3(x1, y0, z1), Vec3(x0, y0, z1), Vec3(0, -1, 0), 0, 0, dx * us, dz * vs);
    glEnd();
}

// 3D smooth cylindrical tube/pipe between two points with lighting normals
static void drawPipe(const Vec3& p0, const Vec3& p1, float radius, int segs = 8) {
    Vec3 d = p1 - p0;
    float len = d.length();
    if (len < 0.0001f) return;
    d = d * (1.0f / len);
    Vec3 up = (std::fabs(d.y) < 0.9f) ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
    Vec3 u = cross(d, up).normalized();
    Vec3 v = cross(d, u).normalized();

    glBegin(GL_QUAD_STRIP);
    for (int i = 0; i <= segs; ++i) {
        float a = 2.0f * PI * (float)i / (float)segs;
        float ca = std::cos(a), sa = std::sin(a);
        Vec3 n = u * ca + v * sa;
        glNormal3f(n.x, n.y, n.z);
        Vec3 o = n * radius;
        glVertex3f(p0.x + o.x, p0.y + o.y, p0.z + o.z);
        glVertex3f(p1.x + o.x, p1.y + o.y, p1.z + o.z);
    }
    glEnd();
}

// 5x7 text on a vertical plane (unlit). facing = +1 readable from +Z, -1 readable from -Z.
static void drawTextWorld(const char* s, float cx, float y, float z, float px, float facing,
                          float r, float g, float b) {
    glDisable(GL_LIGHTING);
    texOff();
    glColor3f(r, g, b);
    float total = (float)std::strlen(s) * 6.0f * px;
    glBegin(GL_QUADS);
    glNormal3f(0, 0, facing);
    float x = -total * 0.5f;
    for (const char* p = s; *p; ++p) {
        int c = (unsigned char)*p;
        if (c >= 'a' && c <= 'z') c -= 32;
        if (c > 127) c = '?';
        for (int row = 0; row < 7; ++row) {
            uint8_t bits = g_fontRows[c][row];
            for (int col = 0; col < 5; ++col)
                if (bits & (0x10 >> col)) {
                    float x0 = x + col * px, x1 = x0 + px, y0 = y + (6 - row) * px, y1 = y0 + px;
                    if (facing > 0) { glVertex3f(cx + x0, y0, z); glVertex3f(cx + x1, y0, z); glVertex3f(cx + x1, y1, z); glVertex3f(cx + x0, y1, z); }
                    else { glVertex3f(cx - x1, y0, z); glVertex3f(cx - x0, y0, z); glVertex3f(cx - x0, y1, z); glVertex3f(cx - x1, y1, z); }
                }
        }
        x += 6.0f * px;
    }
    glEnd();
    glEnable(GL_LIGHTING);
}

// 5x7 text centered at (0, 0, 0) on the local XY plane, facing +Z.
// charW: width of 1 column pixel, charH: height of 1 row pixel.
// bold: slight horizontal dilation for thick, legible strokes.
static void drawTextLocal(const char* s, float charW, float charH, float r, float g, float b, bool bold = false, float zOffset = 0.0f) {
    if (!s || !*s) return;
    glDisable(GL_LIGHTING);
    texOff();
    glColor3f(r, g, b);
    int len = (int)std::strlen(s);
    float totalW = (float)len * 6.0f * charW;
    float startX = -totalW * 0.5f;
    float startY = -3.5f * charH;

    glBegin(GL_QUADS);
    glNormal3f(0, 0, 1);
    float curX = startX;
    for (const char* p = s; *p; ++p) {
        int c = (unsigned char)*p;
        if (c >= 'a' && c <= 'z') c -= 32;
        if (c > 127 || c < 0) c = '?';
        for (int row = 0; row < 7; ++row) {
            uint8_t bits = g_fontRows[c][row];
            if (!bits) continue;
            float y0 = startY + (6 - row) * charH;
            float y1 = y0 + charH;
            int col = 0;
            while (col < 5) {
                if (bits & (0x10 >> col)) {
                    int cStart = col;
                    while (col < 5 && (bits & (0x10 >> col))) ++col;
                    float x0 = curX + cStart * charW;
                    float x1 = curX + col * charW + (bold ? charW * 0.35f : 0.0f);
                    glVertex3f(x0, y0, zOffset);
                    glVertex3f(x1, y0, zOffset);
                    glVertex3f(x1, y1, zOffset);
                    glVertex3f(x0, y1, zOffset);
                } else {
                    ++col;
                }
            }
        }
        curX += 6.0f * charW;
    }
    glEnd();
    glEnable(GL_LIGHTING);
}

// Left-aligned or right-aligned variant on local XY plane, facing +Z.
// align: -1 left, 0 center, +1 right
static void drawTextLocalAlign(const char* s, float startX, float startY, float charW, float charH, float r, float g, float b, int align = -1, bool bold = false, float zOffset = 0.0f) {
    if (!s || !*s) return;
    glDisable(GL_LIGHTING);
    texOff();
    glColor3f(r, g, b);
    int len = (int)std::strlen(s);
    float totalW = (float)len * 6.0f * charW;
    float ox = startX;
    if (align == 0) ox -= totalW * 0.5f;
    else if (align > 0) ox -= totalW;
    float oy = startY - 3.5f * charH;

    glBegin(GL_QUADS);
    glNormal3f(0, 0, 1);
    float curX = ox;
    for (const char* p = s; *p; ++p) {
        int c = (unsigned char)*p;
        if (c >= 'a' && c <= 'z') c -= 32;
        if (c > 127 || c < 0) c = '?';
        for (int row = 0; row < 7; ++row) {
            uint8_t bits = g_fontRows[c][row];
            if (!bits) continue;
            float y0 = oy + (6 - row) * charH;
            float y1 = y0 + charH;
            int col = 0;
            while (col < 5) {
                if (bits & (0x10 >> col)) {
                    int cStart = col;
                    while (col < 5 && (bits & (0x10 >> col))) ++col;
                    float x0 = curX + cStart * charW;
                    float x1 = curX + col * charW + (bold ? charW * 0.35f : 0.0f);
                    glVertex3f(x0, y0, zOffset);
                    glVertex3f(x1, y0, zOffset);
                    glVertex3f(x1, y1, zOffset);
                    glVertex3f(x0, y1, zOffset);
                } else {
                    ++col;
                }
            }
        }
        curX += 6.0f * charW;
    }
    glEnd();
    glEnable(GL_LIGHTING);
}

// ------------------------------------------------------------------ sky ----
static void buildSkyList() {
    if (g_listSky) glDeleteLists(g_listSky, 1);
    g_listSky = glGenLists(1);
    const ClimateProfile& cp = CLIMATES[g_climate];
    auto skyColor = [&](float e, float* c) {
        if (e <= 0.0f) { for (int i = 0; i < 3; ++i) c[i] = cp.fogColor[i] * 0.97f; return; }
        float a = smooth01(e / 0.16f);
        float k = smooth01(std::pow(e / (PI * 0.5f), 0.55f));
        float mid[3];
        mixColor(cp.fogColor, cp.skyHorizon, a, mid);
        mixColor(mid, cp.skyTop, k, c);
    };
    glNewList(g_listSky, GL_COMPILE);
    const int rings = 26, segs = 40;
    const float R = 3000.0f;
    for (int r = 0; r < rings; ++r) {
        float t0 = (float)r / rings, t1 = (float)(r + 1) / rings;
        float e0 = -0.25f + (PI * 0.5f + 0.25f) * std::pow(t0, 1.8f);
        float e1 = -0.25f + (PI * 0.5f + 0.25f) * std::pow(t1, 1.8f);
        float c0[3], c1[3];
        skyColor(e0, c0); skyColor(e1, c1);
        glBegin(GL_QUAD_STRIP);
        for (int s = 0; s <= segs; ++s) {
            float az = 2.0f * PI * s / segs;
            glColor3fv(c0); glVertex3f(std::cos(e0) * std::sin(az) * R, std::sin(e0) * R, std::cos(e0) * std::cos(az) * R);
            glColor3fv(c1); glVertex3f(std::cos(e1) * std::sin(az) * R, std::sin(e1) * R, std::cos(e1) * std::cos(az) * R);
        }
        glEnd();
    }
    glEndList();
}

// -------------------------------------------------------------- terrain ----
static void terrainShade(float x, float z, float* col, Vec3& n) {
    const ClimateProfile& cp = CLIMATES[g_climate];
    float e = TERRAIN_STEP;
    float hx = (terrainHeight(x + e, z) - terrainHeight(x - e, z)) / (2 * e);
    float hz = (terrainHeight(x, z + e) - terrainHeight(x, z - e)) / (2 * e);
    n = Vec3(-hx, 1.0f, -hz).normalized();
    float h = terrainHeight(x, z);
    float slope = 1.0f - n.y;
    float rm = std::max(smooth01((slope - 0.14f) / 0.22f), 0.8f * smooth01((h - 30.0f) / 60.0f));
    float var = 0.80f + 0.40f * fbmTile((x + TERRAIN_HALF) / (2 * TERRAIN_HALF), (z + TERRAIN_HALF) / (2 * TERRAIN_HALF), 14, 14, 3, 7);
    for (int i = 0; i < 3; ++i) col[i] = lerpf(cp.ground[i], cp.rock[i], rm) * var;
    if (!cp.snow) {
        float cap = smooth01((h - 135.0f) / 35.0f);
        for (int i = 0; i < 3; ++i) col[i] = lerpf(col[i], 0.93f, cap);
    }
    float d = terrainRoadDist(x, z);
    float dd = smooth01(1.0f - (d - 8.0f) / 30.0f) * 0.55f;
    for (int i = 0; i < 3; ++i) col[i] = lerpf(col[i], cp.roadTint[i] * 0.78f, dd);
}

static void buildTerrainList() {
    if (g_listTerrain) glDeleteLists(g_listTerrain, 1);
    g_listTerrain = glGenLists(1);
    std::vector<float> col((size_t)HF_N * HF_N * 3);
    std::vector<Vec3> nrm((size_t)HF_N * HF_N);
    for (int iz = 0; iz < HF_N; ++iz)
        for (int ix = 0; ix < HF_N; ++ix) {
            float x = -TERRAIN_HALF + ix * TERRAIN_STEP, z = -TERRAIN_HALF + iz * TERRAIN_STEP;
            size_t k = (size_t)iz * HF_N + ix;
            terrainShade(x, z, &col[k * 3], nrm[k]);
        }
    glNewList(g_listTerrain, GL_COMPILE);
    texOn(g_texGround);
    for (int iz = 0; iz < HF_N - 1; ++iz) {
        glBegin(GL_QUAD_STRIP);
        for (int ix = 0; ix < HF_N; ++ix) {
            for (int dz = 1; dz >= 0; --dz) {
                int jz = iz + dz;
                size_t k = (size_t)jz * HF_N + ix;
                float x = -TERRAIN_HALF + ix * TERRAIN_STEP, z = -TERRAIN_HALF + jz * TERRAIN_STEP;
                glColor3fv(&col[k * 3]);
                glN(nrm[k]);
                glTexCoord2f(x / 14.0f, z / 14.0f);
                glVertex3f(x, g_hf[k], z);
            }
        }
        glEnd();
    }
    glEndList();
}

// ----------------------------------------------------------------- road ----
static void buildRoadList() {
    if (g_listRoad) glDeleteLists(g_listRoad, 1);
    g_listRoad = glGenLists(1);
    const ClimateProfile& cp = CLIMATES[g_climate];
    int N = (int)g_nodes.size();
    if (N < 2) return;
    int limit = g_stageClosed ? N : N - 1;
    auto nodeAt = [&](int i) -> const StageNode& { return g_nodes[i % N]; };
    auto distAt = [&](int i) -> float { return i < N ? g_nodes[i].dist : g_stageLength; };

    glNewList(g_listRoad, GL_COMPILE);

    // 1. sloped banks (shoulders): road dust colour at the curb fading into the terrain colour
    texOn(g_texGround);
    for (int side = -1; side <= 1; side += 2) {
        glBegin(GL_QUAD_STRIP);
        for (int i = 0; i <= limit; ++i) {
            const StageNode& nd = nodeAt(i);
            float w0 = nd.roadWidth * 0.5f + CURB_W, w1 = w0 + SHOULDER_W;
            Vec3 pin = nd.pos + nd.right * (side * w0);
            Vec3 pout = nd.pos + nd.right * (side * w1);
            pin.y = nd.pos.y - 0.005f;
            pout.y = terrainHeight(pout.x, pout.z);
            float cOut[3]; Vec3 nOut;
            terrainShade(pout.x, pout.z, cOut, nOut);
            float cIn[3] = {cp.roadTint[0] * 0.80f, cp.roadTint[1] * 0.80f, cp.roadTint[2] * 0.80f};
            float v = distAt(i) / 14.0f;
            Vec3 n = (nd.up * 3.0f + nd.right * (side * 0.35f)).normalized();
            Vec3 a = side < 0 ? pout : pin, b = side < 0 ? pin : pout;
            const float* ca = side < 0 ? cOut : cIn;
            const float* cb = side < 0 ? cIn : cOut;
            glN(n);
            glColor3fv(ca); glTexCoord2f(0.0f, v); glV(a);
            glColor3fv(cb); glTexCoord2f(1.2f, v); glV(b);
        }
        glEnd();
    }

    // 2. red / white rally curbs
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(-1.0f, -1.0f);
    texOn(g_texCurb);
    glColor3f(1, 1, 1);
    for (int side = -1; side <= 1; side += 2) {
        glBegin(GL_QUAD_STRIP);
        for (int i = 0; i <= limit; ++i) {
            const StageNode& nd = nodeAt(i);
            float w0 = nd.roadWidth * 0.5f, w1 = w0 + CURB_W;
            Vec3 p0 = nd.pos + nd.right * (side * w0) + Vec3(0, 0.035f, 0);
            Vec3 p1 = nd.pos + nd.right * (side * w1) + Vec3(0, 0.095f, 0);
            float v = distAt(i) / 2.0f;
            glN(nd.up);
            Vec3 a = side < 0 ? p1 : p0, b = side < 0 ? p0 : p1;
            glTexCoord2f(0.0f, v); glV(a);
            glTexCoord2f(1.0f, v); glV(b);
        }
        glEnd();
    }

    // 3. main road surface (asphalt/gravel strip)
    texOn(g_texRoad);
    glColor3fv(cp.roadTint);
    glBegin(GL_QUAD_STRIP);
    for (int i = 0; i <= limit; ++i) {
        const StageNode& nd = nodeAt(i);
        float hw = nd.roadWidth * 0.5f;
        Vec3 pl = nd.pos - nd.right * hw + Vec3(0, 0.03f, 0);
        Vec3 pr = nd.pos + nd.right * hw + Vec3(0, 0.03f, 0);
        float v = distAt(i) / 9.0f;
        glN(nd.up);
        glTexCoord2f(0.0f, v); glV(pl);
        glTexCoord2f(1.0f, v); glV(pr);
    }
    glEnd();
    texOff();

    // 4. checkered start / finish lines
    auto checker = [&](int idx) {
        const StageNode& nd = g_nodes[idx];
        float hw = nd.roadWidth * 0.5f;
        const int cols = 12, rows = 2;
        glBegin(GL_QUADS);
        glN(nd.up);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                float c0 = -hw + 2 * hw * c / cols, c1 = -hw + 2 * hw * (c + 1) / cols;
                float f0 = r * 0.7f - 0.7f, f1 = r * 0.7f;
                if ((r + c) & 1) glColor3f(0.05f, 0.05f, 0.05f); else glColor3f(0.95f, 0.95f, 0.95f);
                glV(nd.pos + nd.right * c0 + nd.forward * f0 + Vec3(0, 0.05f, 0));
                glV(nd.pos + nd.right * c1 + nd.forward * f0 + Vec3(0, 0.05f, 0));
                glV(nd.pos + nd.right * c1 + nd.forward * f1 + Vec3(0, 0.05f, 0));
                glV(nd.pos + nd.right * c0 + nd.forward * f1 + Vec3(0, 0.05f, 0));
            }
        glEnd();
    };
    checker(std::min(10, N - 1));
    if (!g_stageClosed) checker(N - 8);
    glDisable(GL_POLYGON_OFFSET_FILL);
    glEndList();
}

// --------------------------------------------------------- scenery models ----
struct RockMesh { std::vector<Vec3> tri; };  // 3 vertices per triangle
static RockMesh g_rockMesh[3];

static void buildRockMeshes() {
    const float t = (1.0f + std::sqrt(5.0f)) * 0.5f;
    Vec3 v[12] = {{-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0}, {0, -1, t}, {0, 1, t},
                  {0, -1, -t}, {0, 1, -t}, {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}};
    int f[20][3] = {{0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4},
                    {11, 10, 2}, {10, 7, 6}, {7, 1, 8}, {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8},
                    {3, 8, 9}, {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}};
    std::vector<Vec3> tris;
    for (auto& fc : f)
        for (int k = 0; k < 3; ++k) tris.push_back(v[fc[k]].normalized());
    for (int level = 0; level < 2; ++level) {
        std::vector<Vec3> next;
        for (size_t i = 0; i < tris.size(); i += 3) {
            Vec3 a = tris[i], b = tris[i + 1], c = tris[i + 2];
            Vec3 ab = ((a + b) * 0.5f).normalized(), bc = ((b + c) * 0.5f).normalized(), ca = ((c + a) * 0.5f).normalized();
            Vec3 t4[12] = {a, ab, ca, ab, b, bc, ca, bc, c, ab, bc, ca};
            for (const Vec3& p : t4) next.push_back(p);
        }
        tris.swap(next);
    }
    for (int var = 0; var < 3; ++var) {
        g_rockMesh[var].tri.clear();
        uint32_t seed = 200u + var * 31u;
        for (const Vec3& d : tris) {
            float r = 0.78f + 0.46f * vnoise3(d.x * 1.7f + var * 5.1f, d.y * 1.7f, d.z * 1.7f, seed) +
                      0.18f * vnoise3(d.x * 4.3f, d.y * 4.3f + var, d.z * 4.3f, seed + 5u);
            Vec3 p(d.x * r * 1.15f, d.y * r * 0.72f, d.z * r * 0.95f);
            p.y = std::max(p.y, -0.35f);
            g_rockMesh[var].tri.push_back(p);
        }
    }
}

static void drawBlob(const Vec3& c, float rad, uint32_t seed, const float* col, float dark) {
    const int st = 6, sl = 9;
    auto vert = [&](int i, int j, Vec3& p, Vec3& n) {
        float phi = PI * i / st;                  // 0 top .. PI bottom
        float th = 2.0f * PI * (j % sl) / sl;
        Vec3 d(std::sin(phi) * std::cos(th), std::cos(phi), std::sin(phi) * std::sin(th));
        float r = rad * (0.82f + 0.30f * hashf(i * 7 + (int)seed, j % sl, seed));
        if (i == 0 || i == st) r = rad * 0.9f;
        p = c + d * r; n = d;
    };
    for (int i = 0; i < st; ++i) {
        glBegin(GL_QUAD_STRIP);
        for (int j = 0; j <= sl; ++j)
            for (int k = 0; k < 2; ++k) {
                Vec3 p, n;
                vert(i + k, j, p, n);
                float shade = (1.0f - dark) + dark * (0.5f + 0.5f * n.y);
                glColor3f(col[0] * shade * 1.15f, col[1] * shade * 1.15f, col[2] * shade * 1.15f);
                glN(n);
                glTexCoord2f(j * 0.5f, (i + k) * 0.5f);
                glV(p);
            }
        glEnd();
    }
}

static void drawTrunk(float r0, float r1, float y0, float y1, int sides) {
    texOn(g_texBark);
    glColor3f(0.85f, 0.78f, 0.70f);
    glBegin(GL_QUAD_STRIP);
    for (int i = 0; i <= sides; ++i) {
        float a = 2.0f * PI * i / sides, ca = std::cos(a), sa = std::sin(a);
        // Organic root flare buttress at base
        float flare = 1.0f + 0.38f * std::cos(3.0f * a);
        glNormal3f(ca, 0.1f, sa);
        glTexCoord2f(i * 1.0f / sides * 2.0f, 0.0f); glVertex3f(ca * r0 * flare, y0, sa * r0 * flare);
        glTexCoord2f(i * 1.0f / sides * 2.0f, (y1 - y0) * 0.5f); glVertex3f(ca * r1, y1, sa * r1);
    }
    glEnd();
    texOff();
}

static void drawDetailedConiferTree(const SceneryObject& o, bool snow) {
    glPushMatrix();
    glTranslatef(o.pos.x, o.pos.y, o.pos.z);
    glRotatef(o.yaw, 0, 1, 0);
    glScalef(o.scale, o.scale * (0.92f + 0.08f * o.variant), o.scale);
    drawTrunk(0.42f, 0.15f, -0.6f, 3.4f, 8);
    texOn(g_texNeedle);
    const int tiers = 7, segs = 14;
    const float snowCol[3] = {0.92f, 0.95f, 1.0f};
    for (int t = 0; t < tiers; ++t) {
        float f = (float)t / (tiers - 1);
        float baseY = 1.25f + t * 1.62f * (1.0f - 0.09f * f);
        float height = 2.85f - 0.20f * t;
        float rad = 2.65f * (1.0f - f * 0.66f);
        float apexY = baseY + height;
        float r1 = rad * 0.56f, y1 = baseY + height * 0.40f;
        float rotOff = (t & 1) ? 0.22f : 0.0f;
        float cApex[3], cMid[3], cLow[3];
        for (int i = 0; i < 3; ++i) {
            cApex[i] = o.color[i] * 1.35f; cMid[i] = o.color[i] * 1.0f; cLow[i] = o.color[i] * 0.52f;
        }
        if (snow) {
            mixColor(cApex, snowCol, 0.85f, cApex);
            mixColor(cMid, snowCol, 0.68f, cMid);
            mixColor(cLow, snowCol, 0.20f, cLow);
        }
        glBegin(GL_TRIANGLES);
        for (int i = 0; i < segs; ++i) {
            float a0 = 2.0f * PI * i / segs + rotOff, a1 = 2.0f * PI * (i + 1) / segs + rotOff;
            float aM = 0.5f * (a0 + a1);
            glColor3fv(cApex); glNormal3f(std::cos(aM), 0.8f, std::sin(aM));
            glTexCoord2f(i * 0.5f, 0.0f); glVertex3f(0, apexY, 0);
            glColor3fv(cMid); glNormal3f(std::cos(a0), 0.55f, std::sin(a0));
            glTexCoord2f(i * 0.5f - 0.25f, 1.0f); glVertex3f(std::cos(a0) * r1, y1, std::sin(a0) * r1);
            glNormal3f(std::cos(a1), 0.55f, std::sin(a1));
            glTexCoord2f(i * 0.5f + 0.25f, 1.0f); glVertex3f(std::cos(a1) * r1, y1, std::sin(a1) * r1);
        }
        glEnd();
        glBegin(GL_QUAD_STRIP);
        for (int i = 0; i <= segs; ++i) {
            float a = 2.0f * PI * i / segs + rotOff;
            float jag = (i % 2 == 0) ? 1.0f : 0.80f;
            jag *= 0.90f + 0.20f * hashf(i % segs, t, 5u + (uint32_t)o.variant);
            float rr = rad * jag;
            glColor3fv(cMid); glNormal3f(std::cos(a), 0.5f, std::sin(a));
            glTexCoord2f(i * 0.5f, 1.0f); glVertex3f(std::cos(a) * r1, y1, std::sin(a) * r1);
            glColor3fv(cLow); glNormal3f(std::cos(a), 0.2f, std::sin(a));
            glTexCoord2f(i * 0.5f, 2.2f); glVertex3f(std::cos(a) * rr, baseY - 0.25f * jag, std::sin(a) * rr);
        }
        glEnd();
    }
    texOff();
    glPopMatrix();
}

static void drawBroadleaf(const SceneryObject& o, bool snow) {
    glPushMatrix();
    glTranslatef(o.pos.x, o.pos.y, o.pos.z);
    glRotatef(o.yaw, 0, 1, 0);
    glScalef(o.scale, o.scale, o.scale);
    drawTrunk(0.44f, 0.22f, -0.6f, 3.5f, 8);
    // a few limbs
    glColor3f(0.28f, 0.20f, 0.14f);
    for (int i = 0; i < 4; ++i) {
        float a = i * 1.57f + o.variant;
        glPushMatrix();
        glTranslatef(0, 2.8f + i * 0.35f, 0);
        glRotatef(a * RAD2DEG, 0, 1, 0);
        glRotatef(-46.0f, 0, 0, 1);
        drawBox(-0.07f, 0, -0.07f, 0.07f, 1.9f, 0.07f);
        glPopMatrix();
    }
    texOn(g_texNeedle);
    float col[3] = {o.color[0], o.color[1], o.color[2]};
    // Autumn amber / golden variant
    if (o.variant == 2)      { col[0] = 0.92f; col[1] = 0.65f; col[2] = 0.12f; }
    else if (o.variant == 3) { col[0] = 0.88f; col[1] = 0.42f; col[2] = 0.14f; }
    if (snow) { const float sc[3] = {0.78f, 0.82f, 0.86f}; mixColor(col, sc, 0.65f, col); }
    drawBlob(Vec3(0, 5.2f, 0), 2.3f, 11u + o.variant, col, 0.45f);
    drawBlob(Vec3(1.5f, 4.4f, 0.6f), 1.7f, 23u + o.variant, col, 0.45f);
    drawBlob(Vec3(-1.4f, 4.6f, -0.7f), 1.8f, 31u + o.variant, col, 0.45f);
    drawBlob(Vec3(0.2f, 6.4f, -0.3f), 1.5f, 41u + o.variant, col, 0.40f);
    texOff();
    glPopMatrix();
}

static void drawBush(const SceneryObject& o, bool snow) {
    glPushMatrix();
    glTranslatef(o.pos.x, o.pos.y, o.pos.z);
    glScalef(o.scale, o.scale, o.scale);
    texOn(g_texNeedle);
    float col[3] = {o.color[0] * 0.95f, o.color[1] * 0.95f, o.color[2] * 0.95f};
    if (snow) { const float sc[3] = {0.85f, 0.88f, 0.92f}; mixColor(col, sc, 0.7f, col); }
    drawBlob(Vec3(0, 0.55f, 0), 0.9f, 5u + o.variant, col, 0.5f);
    drawBlob(Vec3(0.7f, 0.4f, 0.3f), 0.6f, 9u + o.variant, col, 0.5f);
    texOff();
    glPopMatrix();
}

static void drawRockBoulder(const SceneryObject& o, bool snow, const float* foliage) {
    glPushMatrix();
    glTranslatef(o.pos.x, o.pos.y, o.pos.z);
    glRotatef(o.yaw, 0, 1, 0);
    glScalef(o.scale, o.scale, o.scale);
    texOn(g_texRock);
    const RockMesh& m = g_rockMesh[o.variant % 3];
    glBegin(GL_TRIANGLES);
    for (size_t i = 0; i + 2 < m.tri.size(); i += 3) {
        const Vec3 &a = m.tri[i], &b = m.tri[i + 1], &c = m.tri[i + 2];
        Vec3 n = cross(b - a, c - a).normalized();
        float shade = 0.70f + 0.45f * hashf((int)i, o.variant, 3u);
        float col[3] = {o.color[0] * shade, o.color[1] * shade, o.color[2] * shade};
        if (snow && n.y > 0.35f) { const float sc[3] = {0.92f, 0.95f, 1.0f}; mixColor(col, sc, smooth01((n.y - 0.35f) / 0.3f), col); }
        else if (!snow && n.y > 0.55f) { float moss[3] = {foliage[0] * 0.9f, foliage[1] * 0.9f, foliage[2] * 0.9f}; mixColor(col, moss, 0.30f, col); }
        glColor3fv(col);
        glN(n);
        float ax = std::fabs(n.x), ay = std::fabs(n.y), az = std::fabs(n.z);
        for (int k = 0; k < 3; ++k) {
            const Vec3& p = m.tri[i + k];
            if (ay >= ax && ay >= az) glTexCoord2f(p.x * 0.7f, p.z * 0.7f);
            else if (ax >= az) glTexCoord2f(p.z * 0.7f, p.y * 0.7f);
            else glTexCoord2f(p.x * 0.7f, p.y * 0.7f);
            glV(p);
        }
    }
    glEnd();
    texOff();
    glPopMatrix();
}

static void drawHayBale(const SceneryObject& o) {
    glPushMatrix();
    glTranslatef(o.pos.x, o.pos.y, o.pos.z);
    glRotatef(o.yaw, 0, 1, 0);
    int layers = o.variant == 1 ? 2 : 1;
    for (int l = 0; l < layers; ++l) {
        float y0 = l * 0.9f - 0.1f, x = l * 0.15f;
        glColor3f(0.86f, 0.72f, 0.30f);
        drawBox(-0.45f + x, y0, -0.7f, 0.45f + x, y0 + 0.9f, 0.7f);
        // Plastic wrap sponsor bands
        glColor3f(0.92f, 0.92f, 0.94f);
        drawBox(-0.455f + x, y0, -0.40f, 0.455f + x, y0 + 0.9f, -0.28f);
        drawBox(-0.455f + x, y0, 0.28f, 0.455f + x, y0 + 0.9f, 0.40f);
        // Red rally barrier stripe
        glColor3f(0.88f, 0.12f, 0.14f);
        drawBox(-0.458f + x, y0 + 0.35f, -0.705f, 0.458f + x, y0 + 0.55f, 0.705f);
    }
    glPopMatrix();
}

// Wall helper: textured quad in metres (tileW x tileH metres per texture repeat)
static void wallQuad(const Vec3& origin, const Vec3& dirU, float width, float height, const Vec3& n,
                     float tileW, float tileH) {
    Vec3 up(0, 1, 0);
    glBegin(GL_QUADS);
    quadTex(origin, origin + dirU * width, origin + dirU * width + up * height, origin + up * height, n,
            0, 0, width / tileW, height / tileH);
    glEnd();
}

static void drawBuilding(const SceneryObject& b, bool snow) {
    glPushMatrix();
    glTranslatef(b.pos.x, b.pos.y, b.pos.z);
    glRotatef(b.yaw, 0, 1, 0);
    float hw = b.w * 0.5f, hd = b.d * 0.5f, h = b.h;
    bool wood = (b.variant == 1 || b.variant == 2);

    // stone foundation reaching below the slope
    texOn(g_texPlaster);
    glColor3f(0.62f, 0.60f, 0.58f);
    drawBox(-hw - 0.08f, -3.5f, -hd - 0.08f, hw + 0.08f, 0.65f, hd + 0.04f, 0.5f, 0.5f);

    // facade walls: 3 symmetric bays on front/back so ground-floor center bay frames the door cleanly
    float tileW = b.w / 3.0f;
    texOn(wood ? g_texFacadeB : g_texFacadeA);
    glColor3fv(b.color);
    if (h > 3.0f) {
        wallQuad(Vec3(-hw, 3.0f, hd), Vec3(1, 0, 0), b.w, h - 3.0f, Vec3(0, 0, 1), tileW, 3.0f);
    }
    wallQuad(Vec3(-hw, 0.0f, hd),           Vec3(1, 0, 0), tileW, 3.0f, Vec3(0, 0, 1), tileW, 3.0f);
    wallQuad(Vec3(hw - tileW, 0.0f, hd),    Vec3(1, 0, 0), tileW, 3.0f, Vec3(0, 0, 1), tileW, 3.0f);
    wallQuad(Vec3(hw, 0, -hd), Vec3(-1, 0, 0), b.w, h, Vec3(0, 0, -1), tileW, 3.0f);
    wallQuad(Vec3(hw, 0, hd), Vec3(0, 0, -1), b.d, h, Vec3(1, 0, 0), 4.0f, 3.0f);
    wallQuad(Vec3(-hw, 0, -hd), Vec3(0, 0, 1), b.d, h, Vec3(-1, 0, 0), 4.0f, 3.0f);
    texOff();

    // center ground-floor entrance bay (matches facade material without window/shutter overlap)
    texOn(wood ? g_texWood : g_texPlaster);
    glColor3fv(b.color);
    wallQuad(Vec3(-hw + tileW, 0.0f, hd), Vec3(1, 0, 0), tileW, std::min(3.0f, h), Vec3(0, 0, 1), 2.0f, 2.0f);
    texOff();

    // front door with frame and step (protruding past foundation so zero z-fighting)
    glColor3f(0.28f, 0.16f, 0.08f);
    drawBox(-0.68f, 0.0f, hd + 0.05f, 0.68f, 2.28f, hd + 0.12f);
    glColor3f(0.84f, 0.82f, 0.76f);
    drawBox(-0.84f, 0.0f, hd + 0.05f, -0.68f, 2.44f, hd + 0.15f);
    drawBox(0.68f, 0.0f, hd + 0.05f, 0.84f, 2.44f, hd + 0.15f);
    drawBox(-0.84f, 2.28f, hd + 0.05f, 0.84f, 2.44f, hd + 0.15f);
    glColor3f(0.55f, 0.54f, 0.52f);
    drawBox(-1.25f, 0.0f, hd + 0.04f, 1.25f, 0.20f, hd + 0.78f);

    float roofCol[3];
    if (b.variant == 0) { roofCol[0] = 0.78f; roofCol[1] = 0.36f; roofCol[2] = 0.26f; }
    else if (b.variant == 1) { roofCol[0] = 0.40f; roofCol[1] = 0.32f; roofCol[2] = 0.26f; }
    else if (b.variant == 2) { roofCol[0] = 0.46f; roofCol[1] = 0.22f; roofCol[2] = 0.18f; }
    else { roofCol[0] = 0.40f; roofCol[1] = 0.40f; roofCol[2] = 0.42f; }
    if (snow) { const float sc[3] = {0.94f, 0.96f, 1.0f}; mixColor(roofCol, sc, 0.88f, roofCol); }

    if (b.variant == 3) {
        // hotel: flat roof with parapet, striped awning sloping downward over the entrance, front sign
        glColor3fv(roofCol);
        texOn(g_texPlaster);
        drawBox(-hw - 0.25f, h, -hd - 0.25f, hw + 0.25f, h + 0.35f, hd + 0.25f, 0.5f, 0.5f);
        texOff();
        glColor3f(0.78f, 0.78f, 0.80f);
        drawBox(-hw - 0.25f, h + 0.35f, hd + 0.15f, hw + 0.25f, h + 1.0f, hd + 0.25f);
        drawBox(-hw - 0.25f, h + 0.35f, -hd - 0.25f, hw + 0.25f, h + 1.0f, -hd - 0.15f);
        drawBox(-hw - 0.25f, h + 0.35f, -hd - 0.25f, -hw - 0.15f, h + 1.0f, hd + 0.25f);
        drawBox(hw + 0.15f, h + 0.35f, -hd - 0.25f, hw + 0.25f, h + 1.0f, hd + 0.25f);
        for (int i = 0; i < 6; ++i) {
            if (i & 1) glColor3f(0.95f, 0.95f, 0.96f); else glColor3f(0.82f, 0.12f, 0.12f);
            float x0 = -1.8f + i * 0.6f;
            glBegin(GL_QUADS);
            glN(Vec3(0, 0.85f, 0.52f));
            glVertex3f(x0, 2.38f, hd + 1.35f); glVertex3f(x0 + 0.6f, 2.38f, hd + 1.35f);
            glVertex3f(x0 + 0.6f, 2.84f, hd + 0.04f); glVertex3f(x0, 2.84f, hd + 0.04f);
            glEnd();
        }
        glColor3f(0.12f, 0.14f, 0.22f);
        drawBox(-2.1f, h + 0.42f, hd + 0.26f, 2.1f, h + 1.15f, hd + 0.36f);
        drawTextWorld("HOTEL", 0.0f, h + 0.55f, hd + 0.38f, 0.075f, 1.0f, 1.0f, 0.88f, 0.25f);
    } else {
        // gable roof (ridge parallel to X) with overhang, tiles, fascia, gable ends, chimney
        float ov = 0.65f;
        float rise = (hd + ov) * (b.variant == 2 ? 0.80f : 0.62f);
        float wallRise = rise * hd / (hd + ov);
        texOn(g_texRoof);
        glColor3fv(roofCol);
        float slopeLen = std::sqrt((hd + ov) * (hd + ov) + rise * rise);
        Vec3 nF = Vec3(0, hd + ov, rise).normalized(), nB = Vec3(0, hd + ov, -rise).normalized();
        glBegin(GL_QUADS);
        quadTex(Vec3(-hw - ov, h - 0.1f, hd + ov), Vec3(hw + ov, h - 0.1f, hd + ov), Vec3(hw + ov, h + rise - 0.1f, 0),
                Vec3(-hw - ov, h + rise - 0.1f, 0), nF, 0, 0, (b.w + 2 * ov) / 2.0f, slopeLen / 2.0f);
        quadTex(Vec3(hw + ov, h - 0.1f, -hd - ov), Vec3(-hw - ov, h - 0.1f, -hd - ov), Vec3(-hw - ov, h + rise - 0.1f, 0),
                Vec3(hw + ov, h + rise - 0.1f, 0), nB, 0, 0, (b.w + 2 * ov) / 2.0f, slopeLen / 2.0f);
        glEnd();
        texOff();
        // gable triangles
        texOn(wood ? g_texWood : g_texPlaster);
        glColor3fv(b.color);
        glBegin(GL_TRIANGLES);
        glN(Vec3(1, 0, 0));
        glTexCoord2f(0, 0); glVertex3f(hw, h, hd); glTexCoord2f(b.d / 2, 0); glVertex3f(hw, h, -hd);
        glTexCoord2f(b.d / 4, wallRise / 2); glVertex3f(hw, h + wallRise, 0);
        glN(Vec3(-1, 0, 0));
        glTexCoord2f(0, 0); glVertex3f(-hw, h, -hd); glTexCoord2f(b.d / 2, 0); glVertex3f(-hw, h, hd);
        glTexCoord2f(b.d / 4, wallRise / 2); glVertex3f(-hw, h + wallRise, 0);
        glEnd();
        texOff();
        // fascia + ridge
        glColor3f(0.28f, 0.20f, 0.14f);
        drawBox(-hw - ov - 0.03f, h - 0.2f, hd + ov - 0.05f, hw + ov + 0.03f, h + 0.05f, hd + ov + 0.05f);
        drawBox(-hw - ov - 0.03f, h - 0.2f, -hd - ov - 0.05f, hw + ov + 0.03f, h + 0.05f, -hd - ov + 0.05f);
        glColor3fv(roofCol);
        drawBox(-hw - ov, h + rise - 0.2f, -0.12f, hw + ov, h + rise - 0.02f, 0.12f);
        // chimney
        texOn(g_texPlaster);
        glColor3f(0.62f, 0.34f, 0.28f);
        float cx = hw * 0.45f, cz = -hd * 0.30f;
        float cy0 = h + rise * (1.0f - std::fabs(cz) / (hd + ov)) - 0.6f;
        drawBox(cx - 0.35f, cy0, cz - 0.35f, cx + 0.35f, h + rise + 1.0f, cz + 0.35f, 1.0f, 1.0f);
        texOff();
        glColor3f(0.22f, 0.22f, 0.24f);
        drawBox(cx - 0.45f, h + rise + 1.0f, cz - 0.45f, cx + 0.45f, h + rise + 1.12f, cz + 0.45f);
        if (snow) {
            glColor3f(0.94f, 0.96f, 1.0f);
            drawBox(cx - 0.42f, h + rise + 1.12f, cz - 0.42f, cx + 0.42f, h + rise + 1.2f, cz + 0.42f);
        }
        // Soft chimney smoke puffs drifting upward
        if (!snow) {
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            texOn(g_texSoft);
            for (int p = 0; p < 4; ++p) {
                float t = std::fmod(g_timeAccum * 0.7f + p * 0.25f, 1.0f);
                float py = h + rise + 1.15f + t * 2.6f;
                float px = cx + t * 0.85f + 0.12f * std::sin(g_timeAccum * 1.5f + p);
                float pz = cz + t * 0.45f;
                float sz = 0.22f + t * 0.58f;
                glColor4f(0.85f, 0.85f, 0.88f, (1.0f - t) * 0.32f);
                glBegin(GL_QUADS);
                glNormal3f(0, 0, 1);
                glTexCoord2f(0, 0); glVertex3f(px - sz, py - sz, pz);
                glTexCoord2f(1, 0); glVertex3f(px + sz, py - sz, pz);
                glTexCoord2f(1, 1); glVertex3f(px + sz, py + sz, pz);
                glTexCoord2f(0, 1); glVertex3f(px - sz, py + sz, pz);
                glEnd();
            }
            texOff();
            glDisable(GL_BLEND);
        }
        if (b.variant == 1) {  // chalet balcony
            glColor3f(0.42f, 0.28f, 0.17f);
            drawBox(-hw * 0.7f, 3.0f, hd, hw * 0.7f, 3.15f, hd + 1.3f);
            drawBox(-hw * 0.7f, 4.0f, hd + 1.2f, hw * 0.7f, 4.08f, hd + 1.3f);
            for (float x = -hw * 0.7f; x <= hw * 0.7f + 0.01f; x += 0.7f) drawBox(x - 0.04f, 3.15f, hd + 1.2f, x + 0.04f, 4.0f, hd + 1.28f);
        }
        if (b.variant == 2) {  // barn doors
            glColor3f(0.32f, 0.20f, 0.12f);
            drawBox(-1.8f, 0.0f, hd + 0.02f, 1.8f, 2.8f, hd + 0.08f);
            glColor3f(0.82f, 0.78f, 0.70f);
            drawBox(-0.04f, 0.0f, hd + 0.07f, 0.04f, 2.8f, hd + 0.12f);
        }
    }
    // Zinc corner drainpipes running from roof eaves down to foundation
    glColor3f(0.44f, 0.46f, 0.50f);
    drawBox(-hw - 0.05f, 0.0f,  hd + 0.02f, -hw + 0.03f, h,  hd + 0.08f);
    drawBox( hw - 0.03f, 0.0f,  hd + 0.02f,  hw + 0.05f, h,  hd + 0.08f);
    drawBox(-hw - 0.05f, 0.0f, -hd - 0.08f, -hw + 0.03f, h, -hd - 0.02f);
    drawBox( hw - 0.03f, 0.0f, -hd - 0.08f,  hw + 0.05f, h, -hd - 0.02f);
    glPopMatrix();
}

static void drawGrandstand(const SceneryObject& g) {
    glPushMatrix();
    glTranslatef(g.pos.x, g.pos.y, g.pos.z);
    glRotatef(g.yaw, 0, 1, 0);
    float hw = g.w * 0.5f, hd = g.d * 0.5f;
    const int tiers = 5;
    float tierD = g.d / tiers;
    texOn(g_texPlaster);
    for (int t = 0; t < tiers; ++t) {
        float zF = hd - t * tierD, zB = zF - tierD;
        float yT = 0.3f + (t + 1) * 0.95f;
        glColor3f(0.60f, 0.60f, 0.62f);
        drawBox(-hw, -1.0f, zB, hw, yT, zF, 0.4f, 0.4f);
    }
    texOff();
    // spectators with different jackets and heads
    for (int t = 0; t < tiers; ++t) {
        float zF = hd - t * tierD, yT = 0.3f + (t + 1) * 0.95f;
        float sz = zF - tierD * 0.55f;
        // bench
        glColor3f(0.12f, 0.20f, 0.48f);
        drawBox(-hw + 0.3f, yT, sz - 0.35f, hw - 0.3f, yT + 0.35f, sz - 0.1f);
        int idx = 0;
        for (float x = -hw + 0.7f; x < hw - 0.5f; x += 0.85f, ++idx) {
            uint32_t h = (uint32_t)(hashf(idx, t, 77u + g.variant) * 4096.0f);
            if ((h & 7) == 0) continue;
            float r = 0.25f + 0.7f * ((h >> 3 & 7) / 7.0f), gg = 0.2f + 0.7f * ((h >> 6 & 7) / 7.0f), b2 = 0.2f + 0.7f * ((h >> 9 & 7) / 7.0f);
            glColor3f(r, gg, b2);
            drawBox(x - 0.22f, yT + 0.2f, sz - 0.18f, x + 0.22f, yT + 0.95f, sz + 0.12f);
            glColor3f(0.80f, 0.62f, 0.50f);
            drawBox(x - 0.12f, yT + 0.95f, sz - 0.1f, x + 0.12f, yT + 1.2f, sz + 0.1f);
            if ((h & 15) == 1) { glColor3f(0.9f, 0.1f, 0.1f); drawBox(x + 0.2f, yT + 0.9f, sz, x + 0.34f, yT + 1.5f, sz + 0.04f); }
        }
    }
    // steel frame and canopy with diagonal roof trusses
    glColor3f(0.20f, 0.22f, 0.26f);
    for (float x = -hw; x <= hw + 0.01f; x += g.w / 4.0f) {
        drawBox(x - 0.1f, -1.0f, -hd - 0.1f, x + 0.1f, 6.8f, -hd + 0.1f);
        drawBox(x - 0.07f, 4.2f, hd - 0.4f, x + 0.07f, 6.3f, hd - 0.26f);
        // diagonal truss brace
        drawBox(x - 0.05f, 4.8f, -hd * 0.5f, x + 0.05f, 6.5f, hd * 0.1f);
    }
    glColor3f(0.78f, 0.80f, 0.84f);
    glBegin(GL_QUADS);
    glN(Vec3(0, 1, 0.2f));
    glVertex3f(-hw - 0.4f, 6.3f, hd + 0.4f); glVertex3f(hw + 0.4f, 6.3f, hd + 0.4f);
    glVertex3f(hw + 0.4f, 6.9f, -hd - 0.3f); glVertex3f(-hw - 0.4f, 6.9f, -hd - 0.3f);
    glEnd();
    // advertising boards (red/white curb texture repeating horizontally along X) and front fascia banner
    texOn(g_texCurb);
    glColor3f(1, 1, 1);
    glBegin(GL_QUADS);
    glNormal3f(0, 0, 1);
    glTexCoord2f(0.0f, 0.0f);         glVertex3f(-hw, 0.0f, hd + 0.18f);
    glTexCoord2f(0.0f, g.w * 0.35f);  glVertex3f( hw, 0.0f, hd + 0.18f);
    glTexCoord2f(1.0f, g.w * 0.35f);  glVertex3f( hw, 1.15f, hd + 0.18f);
    glTexCoord2f(1.0f, 0.0f);         glVertex3f(-hw, 1.15f, hd + 0.18f);
    glEnd();
    texOff();
    // Back windbreak wall + front canopy fascia with WRC / RALLY1 / HYBRID banner
    glColor3f(0.10f, 0.12f, 0.18f);
    drawBox(-hw, 3.5f, -hd - 0.05f, hw, 6.85f, -hd + 0.12f);
    drawBox(-hw - 0.38f, 6.15f, hd + 0.25f, hw + 0.38f, 6.95f, hd + 0.42f);
    const char* names[3] = {"WRC RALLY1", "HYBRID BOOST", "FIA WORLD RALLY"};
    drawTextWorld(names[g.variant % 3], 0.0f, 6.30f, hd + 0.45f, 0.095f, 1.0f, 1.0f, 0.92f, 0.18f);

    // Waving rally flags on canopy roof
    for (float fx : {-hw + 2.0f, 0.0f, hw - 2.0f}) {
        glColor3f(0.85f, 0.86f, 0.90f);
        drawBox(fx - 0.035f, 6.8f, -hd, fx + 0.035f, 8.4f, -hd + 0.07f); // flagpole
        float wave = std::sin(g_timeAccum * 5.5f + fx * 0.4f);
        glDisable(GL_LIGHTING);
        glBegin(GL_QUADS);
        if (fx < -1.0f)      glColor3f(0.88f, 0.12f, 0.14f);
        else if (fx > 1.0f)  glColor3f(0.12f, 0.48f, 0.92f);
        else                 glColor3f(0.96f, 0.84f, 0.14f);
        glVertex3f(fx, 7.55f, -hd);
        glVertex3f(fx + 0.85f, 7.50f, -hd + wave * 0.22f);
        glVertex3f(fx + 0.85f, 8.35f, -hd + wave * 0.22f);
        glVertex3f(fx, 8.40f, -hd);
        glEnd();
        glEnable(GL_LIGHTING);
    }
    glPopMatrix();
}

static void drawArch(const SceneryObject& a) {
    glPushMatrix();
    glTranslatef(a.pos.x, a.pos.y, a.pos.z);
    glRotatef(a.yaw, 0, 1, 0);
    float hw = a.w * 0.5f, h = a.h;
    texOn(g_texCurb);
    glColor3f(1, 1, 1);
    drawBox(-hw - 0.5f, -2.0f, -0.5f, -hw + 0.5f, h, 0.5f, 0.5f, 0.5f);
    drawBox(hw - 0.5f, -2.0f, -0.5f, hw + 0.5f, h, 0.5f, 0.5f, 0.5f);
    texOff();
    glColor3f(0.10f, 0.11f, 0.15f);
    drawBox(-hw - 0.6f, h - 1.9f, -0.4f, hw + 0.6f, h + 0.3f, 0.4f);
    // banner text readable from the driving direction (looking toward +Z => we see the -Z face)
    const char* txt = a.archType == 0 ? "START" : "FINISH";
    drawTextWorld(txt, 0.0f, h - 1.45f, -0.42f, 0.17f, -1.0f, 1.0f, 1.0f, 1.0f);
    drawTextWorld(txt, 0.0f, h - 1.45f, 0.42f, 0.17f, 1.0f, 1.0f, 1.0f, 1.0f);
    // Digital Race Clock display
    glColor3f(0.04f, 0.04f, 0.05f);
    drawBox(-1.8f, h - 0.55f, -0.415f, 1.8f, h - 0.08f, 0.415f);
    drawTextWorld("00:00.00", 0.0f, h - 0.48f, -0.425f, 0.065f, -1.0f, 0.20f, 0.98f, 0.35f);
    drawTextWorld("00:00.00", 0.0f, h - 0.48f,  0.425f, 0.065f,  1.0f, 0.20f, 0.98f, 0.35f);
    // checkered strip on top
    glDisable(GL_LIGHTING);
    glBegin(GL_QUADS);
    int cells = (int)(a.w * 1.2f);
    for (int i = 0; i < cells; ++i)
        for (int r = 0; r < 2; ++r) {
            if ((i + r) & 1) glColor3f(0.05f, 0.05f, 0.05f); else glColor3f(0.95f, 0.95f, 0.95f);
            float x0 = -hw - 0.6f + (a.w + 1.2f) * i / cells, x1 = -hw - 0.6f + (a.w + 1.2f) * (i + 1) / cells;
            float y0 = h + 0.3f + r * 0.35f, y1 = y0 + 0.35f;
            glVertex3f(x0, y0, 0.0f); glVertex3f(x1, y0, 0.0f); glVertex3f(x1, y1, 0.0f); glVertex3f(x0, y1, 0.0f);
        }
    glEnd();
    glEnable(GL_LIGHTING);
    glPopMatrix();
}

static void drawChevron(const SceneryObject& ch) {
    glPushMatrix();
    glTranslatef(ch.pos.x, ch.pos.y, ch.pos.z);
    glRotatef(ch.yaw, 0, 1, 0);

    // Twin sturdy wooden support posts
    glColor3f(0.36f, 0.28f, 0.22f);
    drawBox(-0.75f, -0.6f, -0.05f, -0.65f, 1.45f, 0.05f);
    drawBox( 0.65f, -0.6f, -0.05f,  0.75f, 1.45f, 0.05f);

    // Outer backing plate & frame
    glColor3f(0.16f, 0.18f, 0.20f);
    drawBox(-0.95f, 0.55f, -0.04f, 0.95f, 1.45f, 0.03f);

    // Reflective chevron arrows visible to approaching rally drivers (on both faces)
    glDisable(GL_LIGHTING);
    glBegin(GL_QUADS);
    const int segs = 6;
    float dir = (ch.variant < 0) ? -1.0f : 1.0f;
    for (float faceZ : {-0.042f, 0.042f}) {
        float fSign = (faceZ < 0.0f) ? -1.0f : 1.0f;
        for (int i = 0; i < segs; ++i) {
            if (i % 2 == 0) glColor3f(0.92f, 0.12f, 0.12f);
            else            glColor3f(0.98f, 0.98f, 1.0f);

            float x0 = -0.92f + (1.84f / segs) * i;
            float x1 = -0.92f + (1.84f / segs) * (i + 1);
            float shift = dir * fSign * 0.12f;

            // Bottom half of chevron V
            glVertex3f(x0 - shift, 0.58f, faceZ);
            glVertex3f(x1 - shift, 0.58f, faceZ);
            glVertex3f(x1 + shift, 1.00f, faceZ);
            glVertex3f(x0 + shift, 1.00f, faceZ);

            // Top half of chevron V
            glVertex3f(x0 + shift, 1.00f, faceZ);
            glVertex3f(x1 + shift, 1.00f, faceZ);
            glVertex3f(x1 - shift, 1.42f, faceZ);
            glVertex3f(x0 - shift, 1.42f, faceZ);
        }
    }
    glEnd();
    glEnable(GL_LIGHTING);
    glPopMatrix();
}

static void drawSceneryObject(const SceneryObject& o, bool snow, const ClimateProfile& cp) {
    switch (o.kind) {
        case SCN_CONIFER: drawDetailedConiferTree(o, snow); break;
        case SCN_BROADLEAF: drawBroadleaf(o, snow); break;
        case SCN_BUSH: drawBush(o, snow); break;
        case SCN_ROCK: drawRockBoulder(o, snow, cp.foliage); break;
        case SCN_HAYBALE: drawHayBale(o); break;
        case SCN_BUILDING: drawBuilding(o, snow); break;
        case SCN_GRANDSTAND: drawGrandstand(o); break;
        case SCN_ARCH: drawArch(o); break;
        case SCN_CHEVRON: drawChevron(o); break;
    }
}

// Group scenery into 160 m chunks, each with a blob-shadow list and a solid-geometry list.
static void buildSceneryChunks() {
    for (SceneChunk& c : g_chunks) { glDeleteLists(c.decals, 1); glDeleteLists(c.solids, 1); }
    g_chunks.clear();
    const ClimateProfile& cp = CLIMATES[g_climate];
    const float CH = 160.0f;
    std::vector<std::pair<long long, int>> keyed;
    for (size_t i = 0; i < g_scenery.size(); ++i) {
        int cx = (int)std::floor(g_scenery[i].pos.x / CH), cz = (int)std::floor(g_scenery[i].pos.z / CH);
        keyed.push_back({((long long)(cx + 1000) << 20) | (long long)(cz + 1000), (int)i});
    }
    std::sort(keyed.begin(), keyed.end());
    size_t i = 0;
    while (i < keyed.size()) {
        size_t j = i;
        while (j < keyed.size() && keyed[j].first == keyed[i].first) ++j;
        SceneChunk ch;
        Vec3 mn(1e9f, 0, 1e9f), mx(-1e9f, 0, -1e9f);
        for (size_t k = i; k < j; ++k) {
            const SceneryObject& o = g_scenery[keyed[k].second];
            mn.x = std::min(mn.x, o.pos.x); mn.z = std::min(mn.z, o.pos.z);
            mx.x = std::max(mx.x, o.pos.x); mx.z = std::max(mx.z, o.pos.z);
        }
        ch.center = Vec3(0.5f * (mn.x + mx.x), 0.0f, 0.5f * (mn.z + mx.z));
        ch.radius = 0.5f * std::sqrt((mx.x - mn.x) * (mx.x - mn.x) + (mx.z - mn.z) * (mx.z - mn.z)) + 30.0f;

        ch.decals = glGenLists(1);
        glNewList(ch.decals, GL_COMPILE);
        for (size_t k = i; k < j; ++k) {
            const SceneryObject& o = g_scenery[keyed[k].second];
            float r = 0.0f;
            switch (o.kind) {
                case SCN_CONIFER: r = 2.6f * o.scale; break;
                case SCN_BROADLEAF: r = 3.3f * o.scale; break;
                case SCN_BUSH: r = 1.3f * o.scale; break;
                case SCN_ROCK: r = 2.0f * o.scale; break;
                case SCN_BUILDING: r = 0.5f * std::sqrt(o.w * o.w + o.d * o.d) + 1.5f; break;
                case SCN_GRANDSTAND: r = 9.0f; break;
                case SCN_CHEVRON: r = 0.95f; break;
                default: break;
            }
            if (r <= 0.0f) continue;
            glPushMatrix();
            glTranslatef(o.pos.x + 0.35f * r * 0.5f, o.pos.y + 0.07f, o.pos.z + 0.35f * r * 0.5f);
            glRotatef(o.yaw, 0, 1, 0);
            glColor4f(0, 0, 0, 0.8f);
            float rx = (o.kind == SCN_BUILDING || o.kind == SCN_GRANDSTAND) ? 0.5f * o.w + 1.2f : r;
            float rz = (o.kind == SCN_BUILDING || o.kind == SCN_GRANDSTAND) ? 0.5f * o.d + 1.2f : r;
            glBegin(GL_QUADS);
            glNormal3f(0, 1, 0);
            glTexCoord2f(0, 0); glVertex3f(-rx, 0, -rz);
            glTexCoord2f(1, 0); glVertex3f(rx, 0, -rz);
            glTexCoord2f(1, 1); glVertex3f(rx, 0, rz);
            glTexCoord2f(0, 1); glVertex3f(-rx, 0, rz);
            glEnd();
            glPopMatrix();
        }
        glEndList();

        ch.solids = glGenLists(1);
        glNewList(ch.solids, GL_COMPILE);
        for (size_t k = i; k < j; ++k) drawSceneryObject(g_scenery[keyed[k].second], cp.snow, cp);
        glEndList();
        g_chunks.push_back(ch);
        i = j;
    }
}

static void buildWorldLists() {
    buildSkyList();
    buildTerrainList();
    buildRoadList();
    buildSceneryChunks();
}

// ============================================================================
//  WRC RALLY1 CAR MODEL (lofted smooth body, flared arches, wing, wheels)
// ============================================================================
static constexpr float WHEEL_R = 0.34f, WHEEL_X = 0.86f, AXLE_F = 1.28f, AXLE_R = -1.27f;
static constexpr float ARCH_R = 0.38f;

static inline void quadCCW(const Vec3& v0, const Vec3& v1, const Vec3& v2, const Vec3& v3, const Vec3& norm) {
    glNormal3f(norm.x, norm.y, norm.z);
    glVertex3f(v0.x, v0.y, v0.z);
    glVertex3f(v1.x, v1.y, v1.z);
    glVertex3f(v2.x, v2.y, v2.z);
    glVertex3f(v3.x, v3.y, v3.z);
}

static inline void quadSymm(float side,
                            const Vec3& v0, const Vec3& v1, const Vec3& v2, const Vec3& v3,
                            Vec3 norm) {
    norm.x *= side;
    glNormal3f(norm.x, norm.y, norm.z);
    glBegin(GL_QUADS);
    if (side > 0) {
        glVertex3f( v0.x, v0.y, v0.z);
        glVertex3f( v1.x, v1.y, v1.z);
        glVertex3f( v2.x, v2.y, v2.z);
        glVertex3f( v3.x, v3.y, v3.z);
    } else {
        glVertex3f(-v0.x, v0.y, v0.z);
        glVertex3f(-v3.x, v3.y, v3.z);
        glVertex3f(-v2.x, v2.y, v2.z);
        glVertex3f(-v1.x, v1.y, v1.z);
    }
    glEnd();
}

// 1. Curved Cambered Airfoil Element (NACA-style curved profile)
static void drawCurvedAirfoil(float x0, float x1, float zLead, float yLead, float chord, float thick, float camber, float pitchDeg, float gurneyH = 0.0f) {
    const int segs = 12;
    float rad = pitchDeg * (PI / 180.0f);
    float cosP = std::cos(rad), sinP = std::sin(rad);

    float pUp[segs + 1][2], pLo[segs + 1][2]; // [z, y]
    Vec3 nUp[segs + 1], nLo[segs + 1];

    for (int i = 0; i <= segs; ++i) {
        float t = (float)i / segs;
        float xc = (1.0f - std::cos(t * PI)) * 0.5f; // cosine spacing for rounded leading edge
        float lz = -xc * chord;
        float yt = thick * 2.6f * (0.2969f * std::sqrt(std::max(0.0001f, xc)) - 0.1260f * xc - 0.3516f * xc * xc + 0.1807f * xc * xc * xc);
        float yc = camber * (4.0f * xc * (1.0f - xc));

        float yU = yc + yt, yL = yc - yt;
        pUp[i][0] = zLead + lz * cosP - yU * sinP;
        pUp[i][1] = yLead + lz * sinP + yU * cosP;
        pLo[i][0] = zLead + lz * cosP - yL * sinP;
        pLo[i][1] = yLead + lz * sinP + yL * cosP;
    }

    for (int i = 0; i <= segs; ++i) {
        int i0 = std::max(0, i - 1), i1 = std::min(segs, i + 1);
        Vec3 tU(0, pUp[i1][1] - pUp[i0][1], pUp[i1][0] - pUp[i0][0]);
        nUp[i] = cross(Vec3(1, 0, 0), tU).normalized();
        Vec3 tL(0, pLo[i1][1] - pLo[i0][1], pLo[i1][0] - pLo[i0][0]);
        nLo[i] = cross(tL, Vec3(1, 0, 0)).normalized();
    }

    // Upper airfoil skin
    glBegin(GL_QUAD_STRIP);
    for (int i = 0; i <= segs; ++i) {
        glN(nUp[i]);
        glVertex3f(x0, pUp[i][1], pUp[i][0]);
        glVertex3f(x1, pUp[i][1], pUp[i][0]);
    }
    glEnd();

    // Lower airfoil skin
    glBegin(GL_QUAD_STRIP);
    for (int i = 0; i <= segs; ++i) {
        glN(nLo[i]);
        glVertex3f(x1, pLo[i][1], pLo[i][0]);
        glVertex3f(x0, pLo[i][1], pLo[i][0]);
    }
    glEnd();

    // Rounded leading edge
    glBegin(GL_QUAD_STRIP);
    glNormal3f(0, 0, 1);
    glVertex3f(x0, pLo[0][1], pLo[0][0]); glVertex3f(x1, pLo[0][1], pLo[0][0]);
    glVertex3f(x0, pUp[0][1], pUp[0][0]); glVertex3f(x1, pUp[0][1], pUp[0][0]);
    glEnd();

    // Trailing edge & Gurney flap
    if (gurneyH > 0.002f) {
        glBegin(GL_QUAD_STRIP);
        glNormal3f(0, 0, -1);
        glVertex3f(x0, pUp[segs][1], pUp[segs][0]);
        glVertex3f(x1, pUp[segs][1], pUp[segs][0]);
        glVertex3f(x0, pUp[segs][1] + gurneyH, pUp[segs][0]);
        glVertex3f(x1, pUp[segs][1] + gurneyH, pUp[segs][0]);
        glEnd();
    }
}

// 2. Swan-Neck Carbon Pylon
static void drawSwanNeckPylon(float x, float zMount, float yMount, float zWing, float yWing, float thick) {
    const int segs = 7;
    Vec2 p0(zMount, yMount);
    Vec2 p1(zMount + 0.12f, yMount + 0.35f);
    Vec2 p2(zWing + 0.10f, yWing + 0.18f);
    Vec2 p3(zWing, yWing);

    Vec2 pts[segs + 1];
    for (int i = 0; i <= segs; ++i) {
        float t = (float)i / segs, it = 1.0f - t;
        pts[i] = p0 * (it * it * it) + p1 * (3.0f * it * it * t) + p2 * (3.0f * it * t * t) + p3 * (t * t * t);
    }

    float hw = thick * 0.5f;
    float depth = 0.045f;
    glColor3f(0.06f, 0.06f, 0.07f);
    glBegin(GL_QUAD_STRIP);
    for (int i = 0; i <= segs; ++i) {
        glVertex3f(x - hw, pts[i].y, pts[i].x);
        glVertex3f(x + hw, pts[i].y, pts[i].x);
    }
    glEnd();
    glBegin(GL_QUAD_STRIP);
    for (int i = 0; i <= segs; ++i) {
        glVertex3f(x + hw, pts[i].y, pts[i].x - depth);
        glVertex3f(x - hw, pts[i].y, pts[i].x - depth);
    }
    glEnd();
    for (int side = -1; side <= 1; side += 2) {
        glBegin(GL_QUAD_STRIP);
        glNormal3f((float)side, 0, 0);
        for (int i = 0; i <= segs; ++i) {
            glVertex3f(x + side * hw, pts[i].y, pts[i].x);
            glVertex3f(x + side * hw, pts[i].y, pts[i].x - depth);
        }
        glEnd();
    }
}

// 3. Wing Endplate with Lower Aero Shelf
static void drawWingEndplate(float x, float z0, float z1, float y0, float y1, float thick, bool isRight) {
    float hw = thick * 0.5f;
    const int nv = 6;
    float zy[nv][2] = {
        {z0, y0 + 0.08f},
        {z0, y1 - 0.12f},
        {z0 - 0.12f, y1},
        {z1, y1},
        {z1, y0},
        {z0 + 0.15f, y0}
    };

    for (int side = -1; side <= 1; side += 2) {
        glBegin(GL_POLYGON);
        glNormal3f((float)side, 0, 0);
        for (int i = 0; i < nv; ++i) {
            int idx = (side > 0) ? i : (nv - 1 - i);
            glVertex3f(x + side * hw, zy[idx][1], zy[idx][0]);
        }
        glEnd();
    }

    glBegin(GL_QUAD_STRIP);
    for (int i = 0; i <= nv; ++i) {
        int i0 = i % nv;
        glVertex3f(x - hw, zy[i0][1], zy[i0][0]);
        glVertex3f(x + hw, zy[i0][1], zy[i0][0]);
    }
    glEnd();

    // Lower horizontal stabilizing winglet shelf (faces UP +Y)
    glColor3f(0.06f, 0.06f, 0.07f);
    glBegin(GL_QUADS);
    if (isRight) {
        quadCCW(Vec3(x, y0 + 0.06f, z1 + 0.05f), Vec3(x + 0.10f, y0 + 0.06f, z1 + 0.05f),
                Vec3(x + 0.10f, y0 + 0.06f, z0 + 0.10f), Vec3(x, y0 + 0.06f, z0 + 0.10f), Vec3(0, 1, 0));
    } else {
        quadCCW(Vec3(x - 0.10f, y0 + 0.06f, z1 + 0.05f), Vec3(x, y0 + 0.06f, z1 + 0.05f),
                Vec3(x, y0 + 0.06f, z0 + 0.10f), Vec3(x - 0.10f, y0 + 0.06f, z0 + 0.10f), Vec3(0, 1, 0));
    }
    glEnd();
}

// 4. Double-Element WRC Rear Wing
static void drawWrcDoubleElementWing(const float accentCol[3]) {
    for (float px : {-0.34f, 0.34f}) {
        drawSwanNeckPylon(px, -1.86f, 1.05f, -1.94f, 1.45f, 0.038f);
    }
    glColor3f(accentCol[0], accentCol[1], accentCol[2]);
    drawCurvedAirfoil(-0.92f, 0.92f, -1.72f, 1.44f, 0.44f, 0.042f, 0.032f, 14.0f, 0.014f);
    glColor3f(0.08f, 0.08f, 0.10f);
    drawCurvedAirfoil(-0.88f, 0.88f, -1.82f, 1.24f, 0.32f, 0.034f, 0.024f, 22.0f, 0.008f);
    for (float side : {-1.0f, 1.0f}) {
        float ex = side * 0.93f;
        drawWingEndplate(ex, -1.60f, -2.22f, 1.02f, 1.54f, 0.022f, side > 0);
    }
}

// 5. Curved Front Bumper, Nose, Carbon Splitter & Dive Planes
static void drawCurvedFrontBumper(const float bodyCol[3], const float accentCol[3], int model = 0) {
    const int segs = 16;
    const float hw = 0.94f;
    const float zCtr = 2.05f, zSweep = 0.20f;

    auto zCurve = [&](float x) -> float {
        float f = x / hw;
        return zCtr - zSweep * (f * f);
    };

    // Carbon front splitter (wide ground-effect floor)
    glColor3f(0.08f, 0.08f, 0.09f);
    glBegin(GL_QUADS);
    for (int i = 0; i < segs; ++i) {
        float x0 = -hw + (2.0f * hw / segs) * i;
        float x1 = -hw + (2.0f * hw / segs) * (i + 1);
        float zF0 = zCurve(x0) + 0.06f, zF1 = zCurve(x1) + 0.06f;
        float zB0 = zCurve(x0) - 0.22f, zB1 = zCurve(x1) - 0.22f;
        quadCCW(Vec3(x0, 0.13f, zB0), Vec3(x1, 0.13f, zB1), Vec3(x1, 0.13f, zF1), Vec3(x0, 0.13f, zF0), Vec3(0, 1, 0));
        float nx = -2.0f * zSweep * ((x0 + x1) * 0.5f / (hw * hw));
        Vec3 nFront = Vec3(nx, 0, 1.0f).normalized();
        quadCCW(Vec3(x0, 0.09f, zF0), Vec3(x1, 0.09f, zF1), Vec3(x1, 0.13f, zF1), Vec3(x0, 0.13f, zF0), nFront);
    }
    glEnd();

    // Splitter endplate vertical winglets at x = ±hw
    for (float side : {-1.0f, 1.0f}) {
        float sz = zCurve(hw) + 0.02f;
        quadSymm(side, Vec3(hw + 0.02f, 0.09f, sz - 0.22f), Vec3(hw + 0.02f, 0.09f, sz),
                       Vec3(hw + 0.02f, 0.24f, sz - 0.04f), Vec3(hw + 0.02f, 0.24f, sz - 0.22f), Vec3(1, 0, 0));
    }

    // Lower Apron (y = 0.13f .. 0.22f across full width)
    glColor3fv(bodyCol);
    glBegin(GL_QUADS);
    for (int i = 0; i < segs; ++i) {
        float x0 = -hw + (2.0f * hw / segs) * i;
        float x1 = -hw + (2.0f * hw / segs) * (i + 1);
        float z0 = zCurve(x0), z1 = zCurve(x1);
        float nx = -2.0f * zSweep * ((x0 + x1) * 0.5f / (hw * hw));
        Vec3 n = Vec3(nx, 0.35f, 0.94f).normalized();
        quadCCW(Vec3(x0, 0.13f, z0), Vec3(x1, 0.13f, z1), Vec3(x1, 0.22f, z1 - 0.01f), Vec3(x0, 0.22f, z0 - 0.01f), n);
    }
    glEnd();

    // Large Center Radiator Grille Mouth (|x| <= 0.52f, y = 0.22f .. 0.44f)
    glColor3f(0.04f, 0.04f, 0.05f);
    glBegin(GL_QUADS);
    quadCCW(Vec3(-0.52f, 0.22f, zCtr - 0.03f), Vec3( 0.52f, 0.22f, zCtr - 0.03f),
            Vec3( 0.46f, 0.44f, zCtr - 0.07f), Vec3(-0.46f, 0.44f, zCtr - 0.07f), Vec3(0, 0, 1));
    glColor3f(0.55f, 0.58f, 0.62f);
    quadCCW(Vec3(-0.46f, 0.23f, zCtr - 0.12f), Vec3( 0.46f, 0.23f, zCtr - 0.12f),
            Vec3( 0.42f, 0.43f, zCtr - 0.15f), Vec3(-0.42f, 0.43f, zCtr - 0.15f), Vec3(0, 0, 1));
    glEnd();

    glColor3f(0.32f, 0.34f, 0.38f);
    glLineWidth(1.5f);
    glBegin(GL_LINES);
    for (float iy = 0.25f; iy <= 0.42f; iy += 0.025f) {
        glVertex3f(-0.44f, iy, zCtr - 0.115f);
        glVertex3f( 0.44f, iy, zCtr - 0.115f);
    }
    glEnd();
    glLineWidth(1.0f);

    // Sparco Rally Fabric Tow Strap
    glColor3f(0.92f, 0.12f, 0.14f);
    glBegin(GL_QUADS);
    quadCCW(Vec3(0.46f, 0.13f, zCtr - 0.01f), Vec3(0.51f, 0.13f, zCtr - 0.01f),
            Vec3(0.51f, 0.24f, zCtr - 0.01f), Vec3(0.46f, 0.24f, zCtr - 0.01f), Vec3(0, 0, 1));
    glColor3f(0.15f, 0.15f, 0.18f);
    quadCCW(Vec3(0.45f, 0.23f, zCtr - 0.015f), Vec3(0.52f, 0.23f, zCtr - 0.015f),
            Vec3(0.52f, 0.26f, zCtr - 0.015f), Vec3(0.45f, 0.26f, zCtr - 0.015f), Vec3(0, 0, 1));
    glEnd();

    // Upper Bumper Center Nose (|x| <= 0.46f, y = 0.44f .. 0.58f)
    glColor3fv(bodyCol);
    glBegin(GL_QUADS);
    const int noseSegs = 8;
    for (int i = 0; i < noseSegs; ++i) {
        float x0 = -0.46f + (0.92f / noseSegs) * i;
        float x1 = -0.46f + (0.92f / noseSegs) * (i + 1);
        float z0 = zCurve(x0), z1 = zCurve(x1);
        float nx = -2.0f * zSweep * ((x0 + x1) * 0.5f / (hw * hw));
        Vec3 n = Vec3(nx, 0.55f, 0.82f).normalized();
        quadCCW(Vec3(x0, 0.44f, z0 - 0.02f), Vec3(x1, 0.44f, z1 - 0.02f),
                Vec3(x1, 0.58f, z1 - 0.06f), Vec3(x0, 0.58f, z0 - 0.06f), n);
    }
    glColor3fv(accentCol);
    quadCCW(Vec3(-0.09f, 0.48f, zCtr - 0.025f), Vec3( 0.09f, 0.48f, zCtr - 0.025f),
            Vec3( 0.07f, 0.56f, zCtr - 0.065f), Vec3(-0.07f, 0.56f, zCtr - 0.065f), Vec3(0, 0.55f, 0.82f));
    glEnd();

    // Front Official Rally Competition License Plate (Left Bumper Mount)
    static const char* frontPlates[3] = {"K-AM 180", "ALZ-WRC 11", "PX23 WRC"};
    {
        float plX = -0.62f, plY = 0.30f;
        glPushMatrix();
        glTranslatef(plX, plY, zCurve(plX) + 0.016f);
        glRotatef(-16.0f, 0, 1, 0); // match curved bumper normal
        glDisable(GL_LIGHTING);

        // Thin black border frame (30cm x 8.5cm)
        glColor3f(0.05f, 0.05f, 0.06f);
        glBegin(GL_QUADS);
        glNormal3f(0, 0, 1);
        glVertex3f(-0.150f, -0.042f, 0.0f);
        glVertex3f( 0.150f, -0.042f, 0.0f);
        glVertex3f( 0.150f,  0.042f, 0.0f);
        glVertex3f(-0.150f,  0.042f, 0.0f);
        glEnd();

        // White plate face
        glColor3f(0.97f, 0.97f, 0.98f);
        glBegin(GL_QUADS);
        glVertex3f(-0.144f, -0.036f, 0.001f);
        glVertex3f( 0.144f, -0.036f, 0.001f);
        glVertex3f( 0.144f,  0.036f, 0.001f);
        glVertex3f(-0.144f,  0.036f, 0.001f);
        glEnd();

        // EU blue strip on left
        glColor3f(0.06f, 0.20f, 0.75f);
        glBegin(GL_QUADS);
        glVertex3f(-0.144f, -0.036f, 0.002f);
        glVertex3f(-0.108f, -0.036f, 0.002f);
        glVertex3f(-0.108f,  0.036f, 0.002f);
        glVertex3f(-0.144f,  0.036f, 0.002f);
        glEnd();

        // Plate registration text in bold black
        glPushMatrix();
        glTranslatef(0.018f, 0.0f, 0.003f);
        drawTextLocal(frontPlates[model % 3], 0.0038f, 0.0064f, 0.06f, 0.06f, 0.08f, true, 0.001f);
        glPopMatrix();

        glEnable(GL_LIGHTING);
        glPopMatrix();
    }

    // Front Nose Rally Brand Badge
    {
        glPushMatrix();
        glTranslatef(0.0f, 0.52f, zCtr - 0.040f);
        glRotatef(20.0f, 1, 0, 0);
        glDisable(GL_LIGHTING);
        if (model == 0) {
            // Toyota Gazoo Racing "GR" emblem
            glColor3f(0.08f, 0.08f, 0.10f);
            drawBox(-0.065f, -0.024f, -0.005f, 0.065f, 0.024f, 0.005f);
            drawTextLocal("GR", 0.0045f, 0.0070f, 0.95f, 0.15f, 0.15f, true);
        } else if (model == 1) {
            // Hyundai N emblem
            glColor3f(0.08f, 0.08f, 0.10f);
            drawBox(-0.055f, -0.024f, -0.005f, 0.055f, 0.024f, 0.005f);
            drawTextLocal("N", 0.0050f, 0.0070f, 0.25f, 0.85f, 0.98f, true);
        } else {
            // Ford ST / M-Sport emblem
            glColor3f(0.08f, 0.08f, 0.10f);
            drawBox(-0.065f, -0.024f, -0.005f, 0.065f, 0.024f, 0.005f);
            drawTextLocal("ST", 0.0045f, 0.0070f, 0.95f, 0.15f, 0.15f, true);
        }
        glEnable(GL_LIGHTING);
        glPopMatrix();
    }

    // Symmetrical Bumper Corners (Left and Right)
    for (float side : {-1.0f, 1.0f}) {
        glColor3fv(bodyCol);
        const int cSegs = 4;
        for (int i = 0; i < cSegs; ++i) {
            float x0 = lerpf(0.52f, hw, (float)i / cSegs);
            float x1 = lerpf(0.52f, hw, (float)(i + 1) / cSegs);
            float z0 = zCurve(x0), z1 = zCurve(x1);
            quadSymm(side, Vec3(x0, 0.22f, z0), Vec3(x1, 0.22f, z1),
                           Vec3(x1, 0.44f, z1 - 0.02f), Vec3(x0, 0.44f, z0 - 0.02f), Vec3(0.45f, 0.1f, 0.88f));
        }

        glColor3f(0.06f, 0.06f, 0.07f);
        float cx0 = 0.58f, cx1 = 0.82f;
        quadSymm(side, Vec3(cx0, 0.24f, zCurve(cx0) - 0.01f), Vec3(cx1, 0.24f, zCurve(cx1) - 0.01f),
                       Vec3(cx1, 0.38f, zCurve(cx1) - 0.03f), Vec3(cx0, 0.38f, zCurve(cx0) - 0.03f), Vec3(0, 0, 1));

        glColor3fv(bodyCol);
        for (int i = 0; i < 2; ++i) {
            float x0 = lerpf(0.86f, hw, (float)i / 2.0f);
            float x1 = lerpf(0.86f, hw, (float)(i + 1) / 2.0f);
            float z0 = zCurve(x0), z1 = zCurve(x1);
            quadSymm(side, Vec3(x0, 0.44f, z0 - 0.02f), Vec3(x1, 0.44f, z1 - 0.02f),
                           Vec3(x1, 0.58f, z1 - 0.06f), Vec3(x0, 0.58f, z0 - 0.06f), Vec3(0.55f, 0.3f, 0.77f));
        }

        glColor3f(0.08f, 0.08f, 0.09f);
        for (float cy : {0.26f, 0.38f}) {
            float kx0 = 0.78f, kx1 = 0.98f;
            float kz0 = zCurve(kx0) + 0.01f, kz1 = zCurve(kx1) - 0.03f;
            quadSymm(side, Vec3(kx0, cy, kz0), Vec3(kx1, cy + 0.03f, kz1),
                           Vec3(kx1, cy - 0.02f, kz1 - 0.14f), Vec3(kx0, cy - 0.04f, kz0 - 0.14f), Vec3(0, 0.94f, -0.34f));
        }
    }
}

// 6. Swept 3D LED Headlights & DRLs
static void drawSwept3DHeadlights() {
    const float hw = 0.94f;
    const float zCtr = 2.05f, zSweep = 0.20f;
    auto zCurve = [&](float x) -> float {
        float f = x / hw;
        return zCtr - zSweep * (f * f);
    };

    for (float side : {-1.0f, 1.0f}) {
        float xIn = 0.46f, xOut = 0.86f;
        float zIn = zCurve(xIn) - 0.03f, zOut = zCurve(xOut) - 0.03f;
        float yBot = 0.44f, yTop = 0.56f;

        glColor3f(0.08f, 0.09f, 0.11f);
        quadSymm(side, Vec3(xIn, yBot, zIn), Vec3(xOut, yBot, zOut),
                       Vec3(xOut, yTop, zOut - 0.02f), Vec3(xIn, yTop, zIn - 0.04f), Vec3(0.35f, 0.3f, 0.88f));

        glDisable(GL_LIGHTING);
        for (float t : {0.35f, 0.75f}) {
            float px = xIn + (xOut - xIn) * t;
            float pz = zIn + (zOut - zIn) * t + 0.005f;
            float py = 0.50f;

            glColor3f(0.85f, 0.88f, 0.92f);
            glBegin(GL_TRIANGLE_FAN);
            glVertex3f(side * px, py, pz);
            for (int k = 0; k <= 10; ++k) {
                float a = (side > 0 ? 1 : -1) * 2.0f * PI * k / 10;
                glVertex3f(side * (px + std::cos(a) * 0.038f), py + std::sin(a) * 0.038f, pz);
            }
            glEnd();

            glColor3f(1.0f, 0.98f, 0.92f);
            glBegin(GL_TRIANGLE_FAN);
            glVertex3f(side * px, py, pz + 0.003f);
            for (int k = 0; k <= 10; ++k) {
                float a = (side > 0 ? 1 : -1) * 2.0f * PI * k / 10;
                glVertex3f(side * (px + std::cos(a) * 0.022f), py + std::sin(a) * 0.022f, pz + 0.003f);
            }
            glEnd();
        }

        glColor3f(0.98f, 1.0f, 1.0f);
        glLineWidth(2.5f);
        glBegin(GL_LINE_STRIP);
        glVertex3f(side * (xIn + 0.02f), yBot + 0.02f, zIn + 0.008f);
        glVertex3f(side * xIn, yTop - 0.015f, zIn + 0.008f);
        glVertex3f(side * (xOut - 0.01f), yTop - 0.015f, zOut + 0.008f);
        glEnd();
        glLineWidth(1.0f);

        glColor3f(0.98f, 0.65f, 0.08f);
        quadSymm(side, Vec3(xOut - 0.04f, yBot + 0.02f, zOut + 0.008f), Vec3(xOut, yBot + 0.02f, zOut + 0.008f),
                       Vec3(xOut, yTop - 0.02f, zOut + 0.008f), Vec3(xOut - 0.04f, yTop - 0.02f, zOut + 0.008f), Vec3(0, 0, 1));

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glColor4f(0.85f, 0.92f, 1.0f, 0.28f);
        quadSymm(side, Vec3(xIn, yBot, zIn + 0.012f), Vec3(xOut, yBot, zOut + 0.012f),
                       Vec3(xOut, yTop, zOut - 0.01f), Vec3(xIn, yTop, zIn - 0.02f), Vec3(0.35f, 0.3f, 0.88f));
        glDisable(GL_BLEND);
        glEnable(GL_LIGHTING);
    }
}

// 7. Sculpted Engine Hood (Flush Seam With Front Bumper & Cowl)
static void drawSculptedHood(const float bodyCol[3], const float accentCol[3]) {
    const int segsX = 14, segsZ = 10;
    const float hwBumper = 0.94f, zNose = 2.05f, sweepNose = 0.20f;
    auto zFrontHood = [&](float x) -> float {
        float f = x / hwBumper;
        return zNose - sweepNose * (f * f) - 0.06f;
    };

    for (int iz = 0; iz < segsZ; ++iz) {
        float tz0 = (float)iz / segsZ, tz1 = (float)(iz + 1) / segsZ;
        float yBase0 = lerpf(0.58f, 0.88f, tz0), yBase1 = lerpf(0.58f, 0.88f, tz1);
        float hw0 = lerpf(0.74f, 0.82f, tz0), hw1 = lerpf(0.74f, 0.82f, tz1);

        for (int ix = 0; ix < segsX; ++ix) {
            float tx0 = (float)ix / segsX, tx1 = (float)(ix + 1) / segsX;
            float f0 = (tx0 - 0.5f) * 2.0f, f1 = (tx1 - 0.5f) * 2.0f;
            float x00 = f0 * hw0, x10 = f1 * hw0;
            float x01 = f0 * hw1, x11 = f1 * hw1;

            float z00 = lerpf(zFrontHood(x00), 0.58f, tz0);
            float z10 = lerpf(zFrontHood(x10), 0.58f, tz0);
            float z01 = lerpf(zFrontHood(x01), 0.58f, tz1);
            float z11 = lerpf(zFrontHood(x11), 0.58f, tz1);

            float d0 = (std::fabs(f0) < 0.45f) ? (0.028f * (1.0f - (f0 / 0.45f) * (f0 / 0.45f))) : 0.0f;
            float d1 = (std::fabs(f1) < 0.45f) ? (0.028f * (1.0f - (f1 / 0.45f) * (f1 / 0.45f))) : 0.0f;

            float y00 = yBase0 + d0 - 0.015f * (f0 * f0 * f0 * f0);
            float y10 = yBase0 + d1 - 0.015f * (f1 * f1 * f1 * f1);
            float y01 = yBase1 + d0 - 0.015f * (f0 * f0 * f0 * f0);
            float y11 = yBase1 + d1 - 0.015f * (f1 * f1 * f1 * f1);

            if (std::fabs((f0 + f1) * 0.5f) < 0.16f) glColor3fv(accentCol);
            else                                     glColor3fv(bodyCol);

            float fMid = (f0 + f1) * 0.5f;
            Vec3 norm = Vec3(-fMid * 0.35f, 0.92f, 0.28f).normalized();

            glBegin(GL_QUADS);
            quadCCW(Vec3(x00, y00, z00), Vec3(x10, y10, z10),
                    Vec3(x11, y11, z11), Vec3(x01, y01, z01), norm);
            glEnd();
        }
    }

    // Dual Recessed Heat Extraction Vents
    for (float side : {-1.0f, 1.0f}) {
        float vx0 = 0.24f, vx1 = 0.46f;
        float vz0 = 1.05f, vz1 = 1.45f;
        float vy = 0.72f;

        glColor3f(0.06f, 0.06f, 0.07f);
        quadSymm(side, Vec3(vx0, vy, vz0), Vec3(vx1, vy, vz0), Vec3(vx1, vy, vz1), Vec3(vx0, vy, vz1), Vec3(0, 1, 0));

        glColor3f(0.12f, 0.12f, 0.14f);
        for (float lz = vz0 + 0.06f; lz <= vz1 - 0.03f; lz += 0.075f) {
            quadSymm(side, Vec3(vx0 + 0.01f, vy, lz), Vec3(vx1 - 0.01f, vy, lz),
                           Vec3(vx1 - 0.01f, vy + 0.022f, lz + 0.035f), Vec3(vx0 + 0.01f, vy + 0.022f, lz + 0.035f), Vec3(0, 0.70f, -0.70f));
        }
    }

    // Flush AeroCatch Latches
    for (float side : {-1.0f, 1.0f}) {
        float lx = 0.58f, lz = 1.55f;
        glColor3f(0.05f, 0.05f, 0.06f);
        quadSymm(side, Vec3(lx - 0.035f, 0.695f, lz - 0.06f), Vec3(lx + 0.035f, 0.695f, lz - 0.06f),
                       Vec3(lx + 0.035f, 0.695f, lz + 0.06f), Vec3(lx - 0.035f, 0.695f, lz + 0.06f), Vec3(0, 1, 0));
        glColor3f(0.85f, 0.85f, 0.88f);
        quadSymm(side, Vec3(lx - 0.012f, 0.702f, lz - 0.03f), Vec3(lx + 0.012f, 0.702f, lz - 0.03f),
                       Vec3(lx + 0.012f, 0.702f, lz + 0.03f), Vec3(lx - 0.012f, 0.702f, lz + 0.03f), Vec3(0, 1, 0));
    }
}

// 8. Greenhouse, Windshield, Roof, Scoop & Windows
static void drawGreenhouseAndRoof(const float bodyCol[3], const float accentCol[3], int model) {
    const int wxSegs = 10, wzSegs = 8;

    for (int iz = 0; iz < wzSegs; ++iz) {
        float tz0 = (float)iz / wzSegs, tz1 = (float)(iz + 1) / wzSegs;
        float z0 = lerpf(0.58f, -0.05f, tz0), z1 = lerpf(0.58f, -0.05f, tz1);
        float y0 = lerpf(0.88f,  1.34f, tz0), y1 = lerpf(0.88f,  1.34f, tz1);
        float hw0 = lerpf(0.74f, 0.64f, tz0), hw1 = lerpf(0.74f, 0.64f, tz1);

        for (int ix = 0; ix < wxSegs; ++ix) {
            float f0 = ((float)ix / wxSegs - 0.5f) * 2.0f;
            float f1 = ((float)(ix + 1) / wxSegs - 0.5f) * 2.0f;
            float x00 = f0 * hw0, x10 = f1 * hw0;
            float x01 = f0 * hw1, x11 = f1 * hw1;
            float cr00 = lerpf(0.042f, 0.024f, tz0) * (1.0f - f0 * f0);
            float cr10 = lerpf(0.042f, 0.024f, tz0) * (1.0f - f1 * f1);
            float cr01 = lerpf(0.042f, 0.024f, tz1) * (1.0f - f0 * f0);
            float cr11 = lerpf(0.042f, 0.024f, tz1) * (1.0f - f1 * f1);

            if (tz1 > 0.78f) {
                glColor3f(accentCol[0] * 0.85f, accentCol[1] * 0.85f, accentCol[2] * 0.85f);
            } else if (std::fabs((f0 + f1) * 0.5f) > 0.88f || tz0 < 0.08f) {
                glColor3f(0.04f, 0.04f, 0.05f);
            } else {
                glColor3f(0.08f, 0.12f, 0.18f);
            }

            float fMid = (f0 + f1) * 0.5f;
            Vec3 norm = Vec3(-fMid * 0.28f, 0.78f, 0.58f).normalized();
            glBegin(GL_QUADS);
            quadCCW(Vec3(x00, y0 + cr00, z0), Vec3(x10, y0 + cr10, z0),
                    Vec3(x11, y1 + cr11, z1), Vec3(x01, y1 + cr01, z1), norm);
            glEnd();
        }
    }

    // Official WRC Windshield Sunstrip Banner (Co-planar with 36-degree windshield rake angle)
    {
        glPushMatrix();
        glTranslatef(0.0f, 1.290f, 0.022f);
        glRotatef(53.86f, 1, 0, 0); // flush with windshield plane
        glDisable(GL_LIGHTING);

        // Dark matte sunstrip field
        if (model == 0)      glColor3f(0.08f, 0.08f, 0.10f); // Toyota dark carbon/black
        else if (model == 1) glColor3f(0.06f, 0.09f, 0.18f); // Hyundai N performance navy
        else                 glColor3f(0.07f, 0.06f, 0.16f); // Ford M-Sport dark navy
        glBegin(GL_QUADS);
        glNormal3f(0, 0, 1);
        glVertex3f(-0.62f, -0.055f, 0.0f);
        glVertex3f( 0.62f, -0.055f, 0.0f);
        glVertex3f( 0.62f,  0.055f, 0.0f);
        glVertex3f(-0.62f,  0.055f, 0.0f);
        glEnd();

        // Team Accent Colored Winglet Ends on Sunstrip
        glColor3fv(accentCol);
        glBegin(GL_QUADS);
        glVertex3f(-0.62f, -0.055f, 0.001f);
        glVertex3f(-0.48f, -0.055f, 0.001f);
        glVertex3f(-0.44f,  0.055f, 0.001f);
        glVertex3f(-0.62f,  0.055f, 0.001f);

        glVertex3f( 0.48f, -0.055f, 0.001f);
        glVertex3f( 0.62f, -0.055f, 0.001f);
        glVertex3f( 0.62f,  0.055f, 0.001f);
        glVertex3f( 0.44f,  0.055f, 0.001f);
        glEnd();

        // Left Brand Badge
        glPushMatrix();
        glTranslatef(-0.510f, 0.0f, 0.003f);
        if (model == 0)      drawTextLocal("GR", 0.0048f, 0.0080f, 0.98f, 0.15f, 0.15f, true);
        else if (model == 1) drawTextLocal("N", 0.0055f, 0.0080f, 0.25f, 0.85f, 0.98f, true);
        else                 drawTextLocal("ST", 0.0048f, 0.0080f, 0.95f, 0.15f, 0.15f, true);
        glPopMatrix();

        // Center Team Championship Title
        static const char* teamNames[3] = {"TOYOTA GAZOO RACING", "HYUNDAI MOTORSPORT", "M-SPORT FORD WRT"};
        glPushMatrix();
        glTranslatef(0.0f, 0.0f, 0.003f);
        drawTextLocal(teamNames[model % 3], 0.0058f, 0.0096f, 1.0f, 1.0f, 1.0f, true);
        glPopMatrix();

        // Right Official WRC Championship Logo Box
        glColor3f(0.06f, 0.18f, 0.72f);
        glBegin(GL_QUADS);
        glVertex3f(0.440f, -0.042f, 0.002f); glVertex3f(0.565f, -0.042f, 0.002f);
        glVertex3f(0.565f,  0.042f, 0.002f); glVertex3f(0.440f,  0.042f, 0.002f);
        glEnd();

        glPushMatrix();
        glTranslatef(0.502f, 0.0f, 0.004f);
        drawTextLocal("WRC", 0.0042f, 0.0070f, 1.0f, 1.0f, 1.0f, true);
        glPopMatrix();

        glEnable(GL_LIGHTING);
        glPopMatrix();
    }

    glColor3fv(bodyCol);
    for (float side : {-1.0f, 1.0f}) {
        quadSymm(side, Vec3(0.74f, 0.88f, 0.58f), Vec3(0.79f, 0.88f, 0.58f),
                       Vec3(0.68f, 1.34f, -0.05f), Vec3(0.63f, 1.34f, -0.05f), Vec3(0.65f, 0.55f, 0.52f));
    }

    const int rzSegs = 8;
    for (int iz = 0; iz < rzSegs; ++iz) {
        float tz0 = (float)iz / rzSegs, tz1 = (float)(iz + 1) / rzSegs;
        float z0 = lerpf(-0.05f, -1.05f, tz0), z1 = lerpf(-0.05f, -1.05f, tz1);
        float hw = 0.64f;

        for (int ix = 0; ix < wxSegs; ++ix) {
            float f0 = ((float)ix / wxSegs - 0.5f) * 2.0f;
            float f1 = ((float)(ix + 1) / wxSegs - 0.5f) * 2.0f;
            float x0 = f0 * hw, x1 = f1 * hw;
            float b0 = 0.024f * std::sin(std::fabs(f0) * PI);
            float b1 = 0.024f * std::sin(std::fabs(f1) * PI);

            if (std::fabs((f0 + f1) * 0.5f) < 0.16f) glColor3fv(accentCol);
            else                                     glColor3fv(bodyCol);

            float fMid = (f0 + f1) * 0.5f;
            Vec3 norm = Vec3(-fMid * 0.22f, 0.98f, 0.0f).normalized();
            glBegin(GL_QUADS);
            quadCCW(Vec3(x0, 1.34f + b0, z0), Vec3(x1, 1.34f + b1, z0),
                    Vec3(x1, 1.34f + b1, z1), Vec3(x0, 1.34f + b0, z1), norm);
            glEnd();
        }
    }

    glColor3fv(accentCol);
    for (int i = 0; i < 8; ++i) {
        float f0 = ((float)i / 8.0f - 0.5f) * 2.0f, f1 = ((float)(i + 1) / 8.0f - 0.5f) * 2.0f;
        float sx0 = f0 * 0.18f, sx1 = f1 * 0.18f;
        glBegin(GL_QUADS);
        quadCCW(Vec3(sx0, 1.36f, -0.16f), Vec3(sx1, 1.36f, -0.16f),
                Vec3(sx1, 1.44f, -0.16f), Vec3(sx0, 1.44f, -0.16f), Vec3(0, 0.85f, 0.38f));
        glEnd();
    }
    glBegin(GL_QUADS);
    quadCCW(Vec3(-0.18f, 1.44f, -0.16f), Vec3( 0.18f, 1.44f, -0.16f),
            Vec3( 0.14f, 1.35f, -0.85f), Vec3(-0.14f, 1.35f, -0.85f), Vec3(0, 0.96f, -0.28f));
    glEnd();
    for (float side : {-1.0f, 1.0f}) {
        quadSymm(side, Vec3(0.18f, 1.36f, -0.16f), Vec3(0.18f, 1.44f, -0.16f),
                       Vec3(0.14f, 1.35f, -0.85f), Vec3(0.14f, 1.35f, -0.85f), Vec3(1, 0, 0));
    }
    glColor3f(0.04f, 0.04f, 0.05f);
    glBegin(GL_QUADS);
    quadCCW(Vec3(-0.16f, 1.365f, -0.155f), Vec3( 0.16f, 1.365f, -0.155f),
            Vec3( 0.16f, 1.430f, -0.155f), Vec3(-0.16f, 1.430f, -0.155f), Vec3(0, 0, 1));
    glEnd();
    glColor3f(0.75f, 0.75f, 0.78f);
    drawBox(-0.008f, 1.365f, -0.16f, 0.008f, 1.430f, -0.12f);

    glColor3f(0.08f, 0.08f, 0.09f);
    glBegin(GL_TRIANGLES);
    quadCCW(Vec3(0.006f, 1.35f, -0.98f), Vec3(0.006f, 1.41f, -1.06f), Vec3(0.006f, 1.35f, -1.08f), Vec3(0.006f, 1.35f, -1.08f), Vec3(1, 0, 0));
    quadCCW(Vec3(-0.006f, 1.35f, -1.08f), Vec3(-0.006f, 1.41f, -1.06f), Vec3(-0.006f, 1.35f, -0.98f), Vec3(-0.006f, 1.35f, -0.98f), Vec3(-1, 0, 0));
    glEnd();

    // 8B. Authentic Curved Rear Windshield (Arka Cam), Ceramic Frit, Heating Lines & High-Mount Stop Light
    const int rxSegs = 10, rzGlassSegs = 8;
    for (int iz = 0; iz < rzGlassSegs; ++iz) {
        float tz0 = (float)iz / (float)rzGlassSegs;
        float tz1 = (float)(iz + 1) / (float)rzGlassSegs;
        float z0 = lerpf(-1.05f, -1.78f, tz0);
        float z1 = lerpf(-1.05f, -1.78f, tz1);
        float y0 = lerpf( 1.33f,  0.92f, tz0);
        float y1 = lerpf( 1.33f,  0.92f, tz1);
        float hw0 = lerpf(0.58f, 0.68f, tz0);
        float hw1 = lerpf(0.58f, 0.68f, tz1);

        for (int ix = 0; ix < rxSegs; ++ix) {
            float f0 = ((float)ix / (float)rxSegs - 0.5f) * 2.0f;
            float f1 = ((float)(ix + 1) / (float)rxSegs - 0.5f) * 2.0f;
            float x00 = f0 * hw0, x10 = f1 * hw0;
            float x01 = f0 * hw1, x11 = f1 * hw1;

            bool border = (std::fabs((f0 + f1) * 0.5f) > 0.82f || tz0 < 0.10f || tz1 > 0.88f);
            if (border) {
                glColor3f(0.04f, 0.04f, 0.05f); // Black ceramic frit border & gasket seal
            } else {
                glColor3f(0.09f, 0.13f, 0.19f); // Tinted glossy automotive glass
            }

            Vec3 norm = Vec3(-((f0 + f1) * 0.5f) * 0.20f, 0.55f, -0.83f).normalized();
            glBegin(GL_QUADS);
            // Exterior facing quad
            quadCCW(Vec3(x11, y1, z1), Vec3(x10, y0, z0),
                    Vec3(x00, y0, z0), Vec3(x01, y1, z1), norm);
            // Interior facing quad (visible when looking back from cockpit)
            quadCCW(Vec3(x01, y1, z1), Vec3(x00, y0, z0),
                    Vec3(x10, y0, z0), Vec3(x11, y1, z1), -norm);
            glEnd();
        }
    }

    // Rear Window Defroster Heating Wires (Arka Cam Rezistans Cizgileri)
    glDisable(GL_LIGHTING);
    glColor3f(0.82f, 0.44f, 0.18f); // Fine copper heating trace
    glLineWidth(1.6f);
    glBegin(GL_LINES);
    for (float tz : {0.26f, 0.38f, 0.50f, 0.62f, 0.74f}) {
        float y = lerpf(1.33f, 0.92f, tz);
        float z = lerpf(-1.05f, -1.78f, tz) - 0.002f;
        float hw = lerpf(0.58f, 0.68f, tz) * 0.78f;
        glVertex3f(-hw, y, z);
        glVertex3f( hw, y, z);
    }
    glEnd();
    glLineWidth(1.0f);

    // High-Mounted 3rd Brake Light Lens at top of rear glass
    glColor3f(0.92f, 0.08f, 0.08f);
    glBegin(GL_QUADS);
    glVertex3f(-0.15f, 1.305f, -1.085f);
    glVertex3f( 0.15f, 1.305f, -1.085f);
    glVertex3f( 0.15f, 1.325f, -1.060f);
    glVertex3f(-0.15f, 1.325f, -1.060f);
    glEnd();
    glEnable(GL_LIGHTING);

    // 8C. Solid Lower Tailgate Deck Panel (Connecting Glass Bottom to Rear Bumper / Lightbar)
    glColor3fv(bodyCol);
    glBegin(GL_QUADS);
    for (int i = 0; i < rxSegs; ++i) {
        float f0 = ((float)i / (float)rxSegs - 0.5f) * 2.0f;
        float f1 = ((float)(i + 1) / (float)rxSegs - 0.5f) * 2.0f;
        float x00 = f0 * 0.68f, x10 = f1 * 0.68f;
        float x01 = f0 * 0.94f, x11 = f1 * 0.94f;
        auto zCurve = [](float x) -> float { float f = x / 0.94f; return -2.06f + 0.18f * (f * f) - 0.03f; };
        float zB0 = zCurve(x01), zB1 = zCurve(x11);
        Vec3 norm(0.0f, 0.45f, -0.89f);
        quadCCW(Vec3(x11, 0.78f, zB1), Vec3(x10, 0.92f, -1.78f),
                Vec3(x00, 0.92f, -1.78f), Vec3(x01, 0.78f, zB0), norm);
    }
    glEnd();

    // Integrated Aerodynamic Tailgate Ducktail Lip in accent color
    glColor3fv(accentCol);
    glBegin(GL_QUADS);
    for (int i = 0; i < rxSegs; ++i) {
        float f0 = ((float)i / (float)rxSegs - 0.5f) * 2.0f;
        float f1 = ((float)(i + 1) / (float)rxSegs - 0.5f) * 2.0f;
        float x0 = f0 * 0.68f, x1 = f1 * 0.68f;
        quadCCW(Vec3(x1, 0.915f, -1.785f), Vec3(x1, 0.932f, -1.775f),
                Vec3(x0, 0.932f, -1.775f), Vec3(x0, 0.915f, -1.785f), Vec3(0, 0.60f, -0.80f));
    }
    glEnd();

    // 8D. C-Pillars / D-Pillars (Sail Panels Flanking Rear Glass)
    glColor3fv(bodyCol);
    for (float side : {-1.0f, 1.0f}) {
        // 1. Upper C-Pillar (Roof corner to Quarter Window trailing edge)
        quadSymm(side, Vec3(0.58f, 1.33f, -1.05f), Vec3(0.68f, 1.05f, -1.40f),
                       Vec3(0.78f, 0.86f, -1.40f), Vec3(0.64f, 1.15f, -1.18f), Vec3(0.70f, 0.50f, -0.50f));
        // 2. Mid C-Pillar (Quarter window trailing edge to rear glass bottom / fender)
        quadSymm(side, Vec3(0.68f, 1.05f, -1.40f), Vec3(0.68f, 0.92f, -1.78f),
                       Vec3(0.78f, 0.80f, -1.65f), Vec3(0.78f, 0.86f, -1.40f), Vec3(0.75f, 0.45f, -0.48f));
        // 3. Lower C-Pillar / Tailgate corner closure (fender top to bumper corner)
        quadSymm(side, Vec3(0.68f, 0.92f, -1.78f), Vec3(0.70f, 0.78f, -1.95f),
                       Vec3(0.88f, 0.78f, -1.88f), Vec3(0.78f, 0.80f, -1.65f), Vec3(0.75f, 0.40f, -0.52f));
    }

    for (float side : {-1.0f, 1.0f}) {
        glColor3f(0.08f, 0.12f, 0.18f);
        quadSymm(side, Vec3(0.77f, 0.90f,  0.52f), Vec3(0.78f, 0.90f, -0.55f),
                       Vec3(0.66f, 1.31f, -0.55f), Vec3(0.67f, 1.31f, -0.02f), Vec3(0.95f, 0.31f, 0.0f));
        quadSymm(side, Vec3(0.78f, 0.90f, -0.58f), Vec3(0.78f, 0.90f, -1.18f),
                       Vec3(0.64f, 1.15f, -1.18f), Vec3(0.66f, 1.31f, -0.58f), Vec3(0.95f, 0.31f, -0.15f));

        glColor3f(0.04f, 0.04f, 0.05f);
        quadSymm(side, Vec3(0.775f, 0.98f, -0.82f), Vec3(0.775f, 0.98f, -1.04f),
                       Vec3(0.755f, 1.08f, -0.98f), Vec3(0.755f, 1.08f, -0.98f), Vec3(1, 0, 0));

        quadSymm(side, Vec3(0.655f, 0.89f, -0.58f), Vec3(0.655f, 0.89f, -0.55f),
                       Vec3(0.665f, 1.32f, -0.55f), Vec3(0.665f, 1.32f, -0.58f), Vec3(1, 0, 0));
    }
}

// 9. Teardrop Aero Side Mirrors
static void drawAeroMirrors(const float bodyCol[3]) {
    for (float side : {-1.0f, 1.0f}) {
        glColor3f(0.08f, 0.08f, 0.09f);
        drawPipe(Vec3(side * 0.78f, 0.90f, 0.44f), Vec3(side * 0.94f, 0.96f, 0.40f), 0.009f, 6);
        drawPipe(Vec3(side * 0.78f, 0.87f, 0.38f), Vec3(side * 0.94f, 0.94f, 0.38f), 0.009f, 6);

        float mx = 0.96f, my = 0.96f, mz = 0.38f;
        glColor3fv(bodyCol);
        glBegin(GL_TRIANGLE_FAN);
        Vec3 nm = Vec3(side, 0.1f, 0.2f);
        glNormal3f(nm.x, nm.y, nm.z);
        glVertex3f(side * (mx + 0.035f), my, mz);
        for (int k = 0; k <= 12; ++k) {
            float a = (side > 0 ? 1 : -1) * 2.0f * PI * (float)k / 12.0f;
            float cy = std::sin(a) * 0.045f, cz = std::cos(a) * 0.075f;
            glVertex3f(side * (mx - 0.025f), my + cy, mz + cz);
        }
        glEnd();

        glColor3f(0.25f, 0.45f, 0.75f);
        quadSymm(side, Vec3(mx + 0.035f, my - 0.038f, mz - 0.076f), Vec3(mx - 0.035f, my - 0.038f, mz - 0.076f),
                       Vec3(mx - 0.035f, my + 0.038f, mz - 0.076f), Vec3(mx + 0.035f, my + 0.038f, mz - 0.076f), Vec3(0, 0, -1));
    }
}

// 10. Flared Widebody Blister Fenders, Side Sills & Sculpted Door Panels
static void drawWidebodySidesAndFenders(const float bodyCol[3], const float accentCol[3]) {
    for (float side : {-1.0f, 1.0f}) {
        // A. Seamless Front Fender Tops (Connecting Hood Edge to Outer Fender Blister)
        const int fzSegs = 8;
        for (int iz = 0; iz < fzSegs; ++iz) {
            float tz0 = (float)iz / (float)fzSegs;
            float tz1 = (float)(iz + 1) / (float)fzSegs;
            float z0 = lerpf(0.58f, 1.80f, tz0);
            float z1 = lerpf(0.58f, 1.80f, tz1);

            float tzH0 = (1.95f - z0) / 1.37f;
            float tzH1 = (1.95f - z1) / 1.37f;
            float hw0 = lerpf(0.82f, 0.74f, 1.0f - tzH0);
            float hw1 = lerpf(0.82f, 0.74f, 1.0f - tzH1);
            float yHood0 = lerpf(0.88f, 0.58f, (z0 - 0.58f) / 1.22f) - 0.015f;
            float yHood1 = lerpf(0.88f, 0.58f, (z1 - 0.58f) / 1.22f) - 0.015f;

            auto flareF = [](float z) -> float {
                if (z >= 0.90f && z <= 1.66f) {
                    return 0.085f * std::sin((z - 0.90f) / 0.76f * PI);
                }
                return 0.0f;
            };
            float xF0 = lerpf(0.92f, 0.88f, (z0 - 0.58f) / 1.22f) + flareF(z0);
            float xF1 = lerpf(0.92f, 0.88f, (z1 - 0.58f) / 1.22f) + flareF(z1);
            float yF0 = yHood0 - 0.016f;
            float yF1 = yHood1 - 0.016f;

            glColor3fv(bodyCol);
            quadSymm(side, Vec3(hw0, yHood0, z0), Vec3(hw1, yHood1, z1),
                           Vec3(xF1, yF1, z1), Vec3(xF0, yF0, z0), Vec3(0.40f, 0.88f, -0.22f));
        }

        // B. Front Fender Forward Corner Panel (ahead of wheel arch, z = 1.66f to front bumper z = 1.85f)
        glColor3fv(bodyCol);
        quadSymm(side, Vec3(0.90f, 0.16f, 1.66f), Vec3(0.94f, 0.16f, 1.85f),
                       Vec3(0.94f, 0.36f, 1.85f), Vec3(0.985f, 0.36f, 1.66f), Vec3(0.88f, 0.22f, 0.42f));
        quadSymm(side, Vec3(0.985f, 0.36f, 1.66f), Vec3(0.94f, 0.36f, 1.85f),
                       Vec3(0.94f, 0.58f, 1.85f), Vec3(0.88f, 0.58f, 1.66f), Vec3(0.88f, 0.35f, 0.32f));

        // C. Front Fender Rear Side Panel & High-Downforce Air Extractor (z = 0.58f to door seam z = 0.90f)
        quadSymm(side, Vec3(0.90f, 0.20f, 0.58f), Vec3(0.90f, 0.20f, 0.90f),
                       Vec3(0.92f, 0.64f, 0.90f), Vec3(0.92f, 0.64f, 0.58f), Vec3(0.96f, 0.26f, 0.0f));
        quadSymm(side, Vec3(0.92f, 0.64f, 0.58f), Vec3(0.92f, 0.64f, 0.90f),
                       Vec3(0.92f, 0.80f, 0.90f), Vec3(0.92f, 0.88f, 0.58f), Vec3(0.85f, 0.52f, 0.0f));

        glColor3f(0.05f, 0.05f, 0.06f);
        quadSymm(side, Vec3(0.89f, 0.26f, 0.82f), Vec3(0.89f, 0.26f, 0.90f),
                       Vec3(0.89f, 0.58f, 0.90f), Vec3(0.89f, 0.58f, 0.82f), Vec3(1, 0, 0));

        // D. Lower Side Sill / Rocker Panel (z = 0.90f to z = -0.89f)
        glColor3f(0.08f, 0.08f, 0.09f);
        quadSymm(side, Vec3(0.84f, 0.12f, -0.89f), Vec3(0.84f, 0.12f, 0.90f),
                       Vec3(0.90f, 0.20f, 0.90f), Vec3(0.90f, 0.20f, -0.89f), Vec3(0.92f, -0.38f, 0.0f));

        glColor3f(0.06f, 0.06f, 0.07f);
        quadSymm(side, Vec3(0.90f, 0.14f, -0.72f), Vec3(0.92f, 0.14f, -0.89f),
                       Vec3(0.92f, 0.32f, -0.89f), Vec3(0.90f, 0.32f, -0.72f), Vec3(1, 0, 0));

        // E. Sculpted Cabin Door Panel (z = 0.58f to z = -0.89f)
        glColor3fv(bodyCol);
        quadSymm(side, Vec3(0.90f, 0.20f, -0.89f), Vec3(0.90f, 0.20f, 0.58f),
                       Vec3(0.92f, 0.64f, 0.58f), Vec3(0.92f, 0.64f, -0.89f), Vec3(0.96f, 0.28f, 0.0f));

        const int dSegs = 6;
        for (int iz = 0; iz < dSegs; ++iz) {
            float t0 = (float)iz / (float)dSegs, t1 = (float)(iz + 1) / (float)dSegs;
            float z0 = lerpf(0.58f, -0.89f, t0), z1 = lerpf(0.58f, -0.89f, t1);
            float yB0 = lerpf(0.88f, 0.90f, t0), yB1 = lerpf(0.88f, 0.90f, t1);
            float xB0 = lerpf(0.79f, 0.78f, t0), xB1 = lerpf(0.79f, 0.78f, t1);
            quadSymm(side, Vec3(0.92f, 0.64f, z1), Vec3(0.92f, 0.64f, z0),
                           Vec3(xB0, yB0, z0), Vec3(xB1, yB1, z1), Vec3(0.86f, 0.51f, 0.0f));
        }

        glColor3fv(accentCol);
        quadSymm(side, Vec3(0.922f, 0.52f, -0.89f), Vec3(0.922f, 0.52f, 0.58f),
                       Vec3(0.922f, 0.62f, 0.58f), Vec3(0.922f, 0.62f, -0.89f), Vec3(1, 0, 0));

        glColor3f(0.04f, 0.04f, 0.05f);
        quadSymm(side, Vec3(0.915f, 0.68f, -0.15f), Vec3(0.915f, 0.68f, -0.05f),
                       Vec3(0.915f, 0.72f, -0.05f), Vec3(0.915f, 0.72f, -0.15f), Vec3(1, 0, 0));

        // F. Rear Quarter Panel (between rear wheel arch z = -1.65f and rear bumper corner z = -1.88f)
        glColor3fv(bodyCol);
        quadSymm(side, Vec3(0.94f, 0.20f, -1.88f), Vec3(0.90f, 0.20f, -1.65f),
                       Vec3(0.92f, 0.64f, -1.65f), Vec3(0.94f, 0.64f, -1.88f), Vec3(0.92f, 0.22f, -0.32f));
        quadSymm(side, Vec3(0.94f, 0.64f, -1.88f), Vec3(0.92f, 0.64f, -1.65f),
                       Vec3(0.78f, 0.80f, -1.65f), Vec3(0.88f, 0.78f, -1.88f), Vec3(0.85f, 0.48f, -0.22f));

        // G. Rear Fender Top (connecting rear quarter window base to rear blister arch)
        const int rSegs = 6;
        for (int iz = 0; iz < rSegs; ++iz) {
            float t0 = (float)iz / (float)rSegs, t1 = (float)(iz + 1) / (float)rSegs;
            float z0 = lerpf(-0.89f, -1.65f, t0), z1 = lerpf(-0.89f, -1.65f, t1);
            float yIn0 = lerpf(0.90f, 0.80f, t0), yIn1 = lerpf(0.90f, 0.80f, t1);
            float flare0 = 0.085f * std::sin(t0 * PI), flare1 = 0.085f * std::sin(t1 * PI);
            float xOut0 = 0.92f + flare0, xOut1 = 0.92f + flare1;
            float yOut = 0.78f;
            quadSymm(side, Vec3(xOut1, yOut, z1), Vec3(xOut0, yOut, z0),
                           Vec3(0.78f, yIn0, z0), Vec3(0.78f, yIn1, z1), Vec3(0.52f, 0.82f, -0.22f));
        }

        // H. Blister Flared Wheel Arches (Front: 1.28f, Rear: -1.27f)
        for (float az : {AXLE_F, AXLE_R}) {
            glColor3f(0.02f, 0.02f, 0.025f);
            glBegin(GL_TRIANGLE_FAN);
            Vec3 normTub = Vec3(side, 0, 0);
            glNormal3f(normTub.x, 0, 0);
            glVertex3f(side * 0.88f, 0.36f, az);
            for (int i = 0; i <= 20; ++i) {
                float a = (side > 0 ? 1 : -1) * PI * (float)i / 20.0f;
                glVertex3f(side * 0.88f, 0.36f + std::sin(a) * 0.37f, az + std::cos(a) * 0.37f);
            }
            glEnd();

            glColor3fv(bodyCol);
            for (int i = 0; i < 20; ++i) {
                float a0 = PI * (float)i / 20.0f, a1 = PI * (float)(i + 1) / 20.0f;
                float cy0 = std::sin(a0), cz0 = std::cos(a0);
                float cy1 = std::sin(a1), cz1 = std::cos(a1);
                Vec3 p0 = Vec3(0.985f, 0.36f + cy0 * 0.38f, az + cz0 * 0.38f);
                Vec3 p1 = Vec3(0.985f, 0.36f + cy1 * 0.38f, az + cz1 * 0.38f);
                Vec3 p2 = Vec3(0.90f,  0.36f + cy1 * 0.46f, az + cz1 * 0.46f);
                Vec3 p3 = Vec3(0.90f,  0.36f + cy0 * 0.46f, az + cz0 * 0.46f);
                Vec3 nFlare = Vec3(0.85f, (cy0 + cy1) * 0.26f, (cz0 + cz1) * 0.26f).normalized();
                quadSymm(side, p0, p1, p2, p3, nFlare);
            }

            glColor3f(0.04f, 0.04f, 0.05f);
            for (int i = 0; i < 20; ++i) {
                float a0 = PI * (float)i / 20.0f, a1 = PI * (float)(i + 1) / 20.0f;
                float cy0 = std::sin(a0), cz0 = std::cos(a0);
                float cy1 = std::sin(a1), cz1 = std::cos(a1);
                Vec3 p0 = Vec3(0.88f,  0.36f + cy0 * 0.38f, az + cz0 * 0.38f);
                Vec3 p1 = Vec3(0.88f,  0.36f + cy1 * 0.38f, az + cz1 * 0.38f);
                Vec3 p2 = Vec3(0.985f, 0.36f + cy1 * 0.38f, az + cz1 * 0.38f);
                Vec3 p3 = Vec3(0.985f, 0.36f + cy0 * 0.38f, az + cz0 * 0.38f);
                quadSymm(side, p0, p1, p2, p3, Vec3(0, -(cy0 + cy1) * 0.5f, -(cz0 + cz1) * 0.5f));
            }
        }

        // I. Front Fender Top Air Extraction Gills
        glColor3f(0.05f, 0.05f, 0.06f);
        for (float fz = AXLE_F - 0.12f; fz <= AXLE_F + 0.14f; fz += 0.07f) {
            quadSymm(side, Vec3(0.88f, 0.77f, fz), Vec3(0.97f, 0.75f, fz),
                           Vec3(0.97f, 0.77f, fz + 0.025f), Vec3(0.88f, 0.79f, fz + 0.025f), Vec3(0.70f, 0.70f, 0));
        }
    }
}

// 11. Curved Rear Bumper, 3D Full-Width LED Lightbar & Diffuser
static void drawRearBumperAndDiffuser(const float bodyCol[3], const float accentCol[3], int model = 0) {
    const int segs = 16;
    const float hw = 0.94f;
    const float zCtr = -2.06f, zSweep = 0.18f;

    auto zCurve = [&](float x) -> float {
        float f = x / hw;
        return zCtr + zSweep * (f * f);
    };

    // Upper Bumper & Tailgate (y = 0.44f .. 0.78f, faces BACKWARD -Z)
    glColor3fv(bodyCol);
    glBegin(GL_QUADS);
    for (int i = 0; i < segs; ++i) {
        float x0 = -hw + (2.0f * hw / segs) * i;
        float x1 = -hw + (2.0f * hw / segs) * (i + 1);
        float z0 = zCurve(x0), z1 = zCurve(x1);
        float nx = -2.0f * zSweep * ((x0 + x1) * 0.5f / (hw * hw));
        Vec3 n = Vec3(nx, 0.35f, -0.92f).normalized();
        quadCCW(Vec3(x1, 0.44f, z1 + 0.01f), Vec3(x0, 0.44f, z0 + 0.01f),
                Vec3(x0, 0.78f, z0 - 0.03f), Vec3(x1, 0.78f, z1 - 0.03f), n);
    }
    glEnd();

    // 3D Full-Width LED Lightbar Housing
    glColor3f(0.06f, 0.06f, 0.08f);
    glBegin(GL_QUADS);
    for (int i = 0; i < segs; ++i) {
        float x0 = -hw + (2.0f * hw / segs) * i;
        float x1 = -hw + (2.0f * hw / segs) * (i + 1);
        float z0 = zCurve(x0), z1 = zCurve(x1);
        quadCCW(Vec3(x1, 0.63f, z1 - 0.025f), Vec3(x0, 0.63f, z0 - 0.025f),
                Vec3(x0, 0.74f, z0 - 0.025f), Vec3(x1, 0.74f, z1 - 0.025f), Vec3(0, 0, -1));
    }
    glEnd();

    glDisable(GL_LIGHTING);
    // Left & Right Tail Light Clusters
    for (float side : {-1.0f, 1.0f}) {
        float x0 = 0.48f, x1 = 0.88f;
        float z0 = zCurve(x0) - 0.028f, z1 = zCurve(x1) - 0.028f;

        glColor3f(1.0f, 0.12f, 0.10f);
        quadSymm(side, Vec3(x1, 0.64f, z1), Vec3(x0, 0.64f, z0),
                       Vec3(x0, 0.73f, z0), Vec3(x1, 0.73f, z1), Vec3(0, 0, -1));

        glColor3f(1.0f, 0.85f, 0.85f);
        glLineWidth(2.2f);
        glBegin(GL_LINE_STRIP);
        glVertex3f(side * x0, 0.72f, z0 - 0.003f);
        glVertex3f(side * x1, 0.72f, z1 - 0.003f);
        glVertex3f(side * x1, 0.65f, z1 - 0.003f);
        glVertex3f(side * (x0 + 0.06f), 0.65f, z0 - 0.003f);
        glEnd();
        glLineWidth(1.0f);
    }

    // Center connecting horizontal LED strip
    glColor3f(0.92f, 0.08f, 0.08f);
    glBegin(GL_QUADS);
    quadCCW(Vec3( 0.48f, 0.675f, zCurve(0.0f) - 0.028f), Vec3(-0.48f, 0.675f, zCurve(0.0f) - 0.028f),
            Vec3(-0.48f, 0.705f, zCurve(0.0f) - 0.028f), Vec3( 0.48f, 0.705f, zCurve(0.0f) - 0.028f), Vec3(0, 0, -1));
    glEnd();

    // 1. Tailgate Model Badging
    glPushMatrix();
    glTranslatef(0.0f, 0.690f, zCurve(0.0f) - 0.025f);
    glRotatef(180.0f, 0, 1, 0);
    glDisable(GL_LIGHTING);
    if (model == 0) {
        // Toyota "GR YARIS" (Red 'GR', white 'YARIS')
        glPushMatrix();
        glTranslatef(-0.065f, 0.0f, 0.0f);
        drawTextLocal("GR", 0.0055f, 0.0085f, 0.95f, 0.12f, 0.12f, true);
        glPopMatrix();
        glPushMatrix();
        glTranslatef(0.035f, 0.0f, 0.0f);
        drawTextLocal("YARIS", 0.0048f, 0.0075f, 0.96f, 0.96f, 0.98f, true);
        glPopMatrix();
    } else if (model == 1) {
        // Hyundai "i20 N"
        drawTextLocal("i20 N", 0.0052f, 0.0080f, 0.28f, 0.82f, 0.98f, true);
    } else {
        // Ford "PUMA WRC"
        drawTextLocal("PUMA WRC", 0.0048f, 0.0075f, 0.25f, 0.55f, 0.98f, true);
    }
    glPopMatrix();

    // 2. Official Rally Competition Rear License Plate
    static const char* rearPlates[3] = {"K-AM 180", "ALZ-WRC 11", "PX23 WRC"};
    glPushMatrix();
    glTranslatef(0.0f, 0.560f, zCurve(0.0f) - 0.022f);
    glRotatef(180.0f, 0, 1, 0);

    // Plate background: European Yellow Rally Plate
    glColor3f(0.96f, 0.82f, 0.14f);
    glBegin(GL_QUADS);
    glNormal3f(0, 0, 1);
    glVertex3f(-0.245f, -0.052f, 0.0f);
    glVertex3f( 0.245f, -0.052f, 0.0f);
    glVertex3f( 0.245f,  0.052f, 0.0f);
    glVertex3f(-0.245f,  0.052f, 0.0f);
    glEnd();

    // Left EU Blue Band
    glColor3f(0.05f, 0.22f, 0.75f);
    glBegin(GL_QUADS);
    glVertex3f(-0.245f, -0.052f, 0.001f);
    glVertex3f(-0.190f, -0.052f, 0.001f);
    glVertex3f(-0.190f,  0.052f, 0.001f);
    glVertex3f(-0.245f,  0.052f, 0.001f);
    glEnd();

    // EU Gold Circle Marker
    glColor3f(0.98f, 0.88f, 0.15f);
    glBegin(GL_QUADS);
    glVertex3f(-0.224f, 0.012f, 0.002f); glVertex3f(-0.211f, 0.012f, 0.002f);
    glVertex3f(-0.211f, 0.025f, 0.002f); glVertex3f(-0.224f, 0.025f, 0.002f);
    glEnd();

    // Registration text: bold black
    glPushMatrix();
    glTranslatef(0.025f, 0.0f, 0.003f);
    drawTextLocal(rearPlates[model % 3], 0.0060f, 0.0092f, 0.05f, 0.05f, 0.08f, true);
    glPopMatrix();

    glEnable(GL_LIGHTING);
    glPopMatrix();

    // Lower accent trim
    glColor3fv(accentCol);
    glBegin(GL_QUADS);
    for (int i = 0; i < segs; ++i) {
        float x0 = -hw + (2.0f * hw / segs) * i;
        float x1 = -hw + (2.0f * hw / segs) * (i + 1);
        float z0 = zCurve(x0), z1 = zCurve(x1);
        quadCCW(Vec3(x1, 0.41f, z1 + 0.01f), Vec3(x0, 0.41f, z0 + 0.01f),
                Vec3(x0, 0.46f, z0 + 0.01f), Vec3(x1, 0.46f, z1 + 0.01f), Vec3(0, 0, -1));
    }
    glEnd();

    // Carbon Diffuser & Venturi Tunnel
    glColor3f(0.07f, 0.07f, 0.08f);
    glBegin(GL_QUADS);
    for (int i = 0; i < segs; ++i) {
        float x0 = -hw + (2.0f * hw / segs) * i;
        float x1 = -hw + (2.0f * hw / segs) * (i + 1);
        float zF0 = zCurve(x0), zF1 = zCurve(x1);
        float zB0 = zCurve(x0) + 0.35f, zB1 = zCurve(x1) + 0.35f;
        quadCCW(Vec3(x1, 0.38f, zF1), Vec3(x0, 0.38f, zF0),
                Vec3(x0, 0.12f, zB0), Vec3(x1, 0.12f, zB1), Vec3(0, -0.65f, -0.75f));
    }
    glEnd();

    // 4 Vertical Diffuser Strakes
    glColor3f(0.05f, 0.05f, 0.06f);
    for (float fx : {-0.62f, -0.22f, 0.22f, 0.62f}) {
        float fz = zCurve(fx);
        glBegin(GL_QUADS);
        float s = (fx > 0) ? 1.0f : -1.0f;
        quadCCW(Vec3(fx, 0.10f, fz - 0.04f), Vec3(fx, 0.10f, fz + 0.36f),
                Vec3(fx, 0.38f, fz + 0.36f), Vec3(fx, 0.38f, fz - 0.04f), Vec3(s, 0, 0));
        glEnd();
    }

    // Dual 3D Cylindrical Exhaust Pipes
    for (float ex : {-0.42f, 0.42f}) {
        float ez = zCurve(ex) - 0.02f, ey = 0.24f;
        const int pSegs = 16;
        float outR = 0.058f, inR = 0.046f, pLen = 0.16f;

        glColor3f(0.68f, 0.70f, 0.74f);
        glBegin(GL_QUAD_STRIP);
        for (int k = 0; k <= pSegs; ++k) {
            float a = 2.0f * PI * k / pSegs, ca = std::cos(a), sa = std::sin(a);
            glNormal3f(ca, sa, 0);
            glVertex3f(ex + ca * outR, ey + sa * outR, ez + pLen);
            glVertex3f(ex + ca * outR, ey + sa * outR, ez);
        }
        glEnd();

        glColor3f(0.10f, 0.10f, 0.12f);
        glBegin(GL_QUAD_STRIP);
        for (int k = 0; k <= pSegs; ++k) {
            float a = 2.0f * PI * k / pSegs, ca = std::cos(a), sa = std::sin(a);
            glNormal3f(-ca, -sa, 0);
            glVertex3f(ex + ca * inR, ey + sa * inR, ez);
            glVertex3f(ex + ca * inR, ey + sa * inR, ez + pLen);
        }
        glEnd();

        glColor3f(0.85f, 0.88f, 0.92f);
        glBegin(GL_QUAD_STRIP);
        for (int k = 0; k <= pSegs; ++k) {
            float a = 2.0f * PI * k / pSegs, ca = std::cos(a), sa = std::sin(a);
            glNormal3f(0, 0, -1);
            glVertex3f(ex + ca * inR,  ey + sa * inR,  ez);
            glVertex3f(ex + ca * outR, ey + sa * outR, ez);
        }
        glEnd();
    }
}

static void buildCarLists() {
    static const float body[3][3] = {{0.95f, 0.95f, 0.96f}, {0.10f, 0.46f, 0.88f}, {0.10f, 0.12f, 0.46f}};
    static const float accent[3][3] = {{0.88f, 0.06f, 0.09f}, {0.95f, 0.24f, 0.12f}, {0.16f, 0.94f, 0.72f}};

    for (int m = 0; m < 3; ++m) {
        if (g_listCarBody[m]) glDeleteLists(g_listCarBody[m], 1);
        g_listCarBody[m] = glGenLists(1);
        glNewList(g_listCarBody[m], GL_COMPILE);

        const float spec[4] = {0.65f, 0.65f, 0.68f, 1.0f};
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, spec);
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 75.0f);

        // 0. Solid Aerodynamic Chassis Belly Pan
        glColor3f(0.06f, 0.06f, 0.07f);
        glBegin(GL_QUADS);
        quadCCW(Vec3(-0.84f, 0.12f, -1.80f), Vec3( 0.84f, 0.12f, -1.80f),
                Vec3( 0.84f, 0.12f,  1.80f), Vec3(-0.84f, 0.12f,  1.80f), Vec3(0, -1, 0));
        glEnd();

        // 1. Sculpted Curved Front Bumper, Nose, Carbon Splitter & Canards
        drawCurvedFrontBumper(body[m], accent[m], m);

        // 2. Swept-Back 3D Aerodynamic LED Headlights & DRL Lightguides
        drawSwept3DHeadlights();

        // 3. Sculpted Hood with Power Bulge, Cooling Louvers & AeroCatches
        drawSculptedHood(body[m], accent[m]);

        // 4. Curved Greenhouse, Windshield Sunstrip, Contoured Roof & Roof Scoop
        drawGreenhouseAndRoof(body[m], accent[m], m);

        // 5. Flared Widebody Blister Fenders, Side Sills & Door Skins
        drawWidebodySidesAndFenders(body[m], accent[m]);

        // 6. Teardrop Aerodynamic Side Mirrors
        drawAeroMirrors(body[m]);

        // 7. Sculpted Rear Tailgate, 3D LED Lightbar & Carbon Diffuser
        drawRearBumperAndDiffuser(body[m], accent[m], m);

        // 8. Double-Element WRC Swan-Neck Rear Wing with Cambered Airfoils
        drawWrcDoubleElementWing(accent[m]);

        // 9. Cockpit Interior: Authentic Tubular FIA Roll Cage (Chrome-Moly Steel)
        glColor3f(0.88f, 0.90f, 0.94f);
        for (float side : {-1.0f, 1.0f}) {
            // A-Pillar main hoop front legs
            drawPipe(Vec3(side * 0.56f, 0.52f, 0.72f), Vec3(side * 0.56f, 1.26f, -0.05f), 0.022f);
            // B-Pillar main hoop vertical legs
            drawPipe(Vec3(side * 0.56f, 0.22f, -0.68f), Vec3(side * 0.56f, 1.28f, -0.68f), 0.024f);
            // Roof longitudinal connecting bars
            drawPipe(Vec3(side * 0.56f, 1.26f, -0.05f), Vec3(side * 0.56f, 1.28f, -0.68f), 0.022f);
            // Rear back stays
            drawPipe(Vec3(side * 0.56f, 1.28f, -0.68f), Vec3(side * 0.52f, 0.42f, -1.65f), 0.022f);
            // Door cross bars (FIA X-bracing)
            drawPipe(Vec3(side * 0.56f, 0.32f, 0.65f), Vec3(side * 0.56f, 0.92f, -0.65f), 0.016f);
            drawPipe(Vec3(side * 0.56f, 0.92f, 0.65f), Vec3(side * 0.56f, 0.32f, -0.65f), 0.016f);
        }
        // Windshield top cross bar
        drawPipe(Vec3(-0.56f, 1.26f, -0.05f), Vec3(0.56f, 1.26f, -0.05f), 0.022f);
        // Main B-pillar hoop top cross bar
        drawPipe(Vec3(-0.56f, 1.28f, -0.68f), Vec3(0.56f, 1.28f, -0.68f), 0.024f);
        // Main hoop diagonal brace
        drawPipe(Vec3(-0.56f, 0.22f, -0.68f), Vec3(0.56f, 1.28f, -0.68f), 0.020f);
        // Rear stay cross bar
        drawPipe(Vec3(-0.52f, 0.88f, -1.15f), Vec3(0.52f, 0.88f, -1.15f), 0.020f);

        // WRC Racing Flat-Bottom Suede Steering Wheel with 12 O'Clock Center Stripe
        Vec3 swCtr(-0.32f, 0.76f, 0.16f);
        const int swSegs = 16;
        for (int k = 0; k < swSegs; ++k) {
            float a0 = 2.0f * PI * (float)k / (float)swSegs;
            float a1 = 2.0f * PI * (float)(k + 1) / (float)swSegs;
            Vec3 p0 = swCtr + Vec3(std::cos(a0) * 0.13f, std::sin(a0) * 0.13f, 0.0f);
            Vec3 p1 = swCtr + Vec3(std::cos(a1) * 0.13f, std::sin(a1) * 0.13f, 0.0f);
            if (k == 3 || k == 4) glColor3f(0.96f, 0.85f, 0.10f); // Yellow center stripe at 12 o'clock
            else                 glColor3f(0.12f, 0.12f, 0.14f); // Charcoal suede
            drawPipe(p0, p1, 0.013f, 6);
        }
        // Steering wheel spokes & center hub
        glColor3f(0.25f, 0.25f, 0.28f);
        drawPipe(swCtr, swCtr + Vec3(-0.13f, 0.0f, 0.0f), 0.008f, 6);
        drawPipe(swCtr, swCtr + Vec3( 0.13f, 0.0f, 0.0f), 0.008f, 6);
        drawPipe(swCtr, swCtr + Vec3( 0.0f, -0.13f, 0.0f), 0.008f, 6);
        // Steering column
        drawPipe(swCtr, swCtr + Vec3(0.0f, -0.10f, 0.25f), 0.020f, 6);

        // Racing Bucket Seats, 6-Point Harnesses & Crew Helmets
        for (float sx : {-0.32f, 0.32f}) {
            glColor3f(0.08f, 0.08f, 0.10f);
            drawBox(sx - 0.19f, 0.22f, -0.65f, sx + 0.19f, 0.36f, -0.16f);
            drawBox(sx - 0.19f, 0.36f, -0.65f, sx + 0.19f, 1.04f, -0.55f);
            drawBox(sx - 0.22f, 0.88f, -0.62f, sx + 0.22f, 1.04f, -0.42f);
            glColor3f(0.88f, 0.12f, 0.14f);
            drawBox(sx - 0.13f, 0.42f, -0.54f, sx - 0.06f, 1.02f, -0.53f);
            drawBox(sx + 0.06f, 0.42f, -0.54f, sx + 0.13f, 1.02f, -0.53f);
            glColor3f(0.95f, 0.95f, 0.96f);
            drawBox(sx - 0.11f, 0.86f, -0.46f, sx + 0.11f, 1.06f, -0.26f);
            glColor3f(0.06f, 0.08f, 0.12f);
            drawBox(sx - 0.095f, 0.94f, -0.27f, sx + 0.095f, 1.01f, -0.245f);
        }

        // Interior Rear Parcel Shelf & Bulkhead (Directly behind bucket seats under rear glass)
        glColor3f(0.08f, 0.08f, 0.10f);
        drawBox(-0.70f, 0.74f, -1.78f, 0.70f, 0.78f, -0.65f);
        drawBox(-0.70f, 0.12f, -0.66f, 0.70f, 0.74f, -0.64f);

        // WRC MoTeC Digital Driver Display Screen (Facing Driver -Z)
        glPushMatrix();
        glTranslatef(-0.32f, 0.75f, 0.176f);
        glRotatef(180.0f, 0, 1, 0);
        glDisable(GL_LIGHTING);

        // Screen bezel & display background (18cm x 11cm)
        glColor3f(0.03f, 0.04f, 0.06f);
        glBegin(GL_QUADS);
        glNormal3f(0, 0, 1);
        glVertex3f(-0.095f, -0.055f, 0.0f);
        glVertex3f( 0.095f, -0.055f, 0.0f);
        glVertex3f( 0.095f,  0.055f, 0.0f);
        glVertex3f(-0.095f,  0.055f, 0.0f);
        glEnd();

        // Top Shift Light LED Bar (Green -> Amber -> Red)
        float ledY = 0.040f;
        for (int l = 0; l < 8; ++l) {
            float lx = -0.075f + l * 0.021f;
            if (l < 3)       glColor3f(0.15f, 0.95f, 0.25f); // Green
            else if (l < 6)  glColor3f(0.98f, 0.85f, 0.10f); // Amber
            else             glColor3f(0.98f, 0.12f, 0.12f); // Red
            glBegin(GL_QUADS);
            glVertex3f(lx, ledY - 0.005f, 0.001f);
            glVertex3f(lx + 0.016f, ledY - 0.005f, 0.001f);
            glVertex3f(lx + 0.016f, ledY + 0.005f, 0.001f);
            glVertex3f(lx, ledY + 0.005f, 0.001f);
            glEnd();
        }

        // Center Gear Indicator (Large Bright Amber "4")
        glPushMatrix();
        glTranslatef(0.005f, 0.005f, 0.002f);
        drawTextLocal("4", 0.0055f, 0.0090f, 0.98f, 0.88f, 0.15f, true);
        glPopMatrix();

        // Speed on left ("138 KM/H")
        glPushMatrix();
        glTranslatef(-0.052f, 0.005f, 0.002f);
        drawTextLocal("138", 0.0035f, 0.0058f, 0.25f, 0.95f, 0.98f, true);
        glPopMatrix();

        // Hybrid boost on right ("HY 98%")
        glPushMatrix();
        glTranslatef(0.052f, 0.005f, 0.002f);
        drawTextLocal("HY 98%", 0.0028f, 0.0048f, 0.20f, 0.85f, 0.40f, true);
        glPopMatrix();

        // Bottom status row ("OIL 92C  RALLY1")
        glPushMatrix();
        glTranslatef(0.0f, -0.034f, 0.002f);
        drawTextLocal("OIL 92C  WRC", 0.0026f, 0.0042f, 0.85f, 0.88f, 0.95f, false);
        glPopMatrix();

        glEnable(GL_LIGHTING);
        glPopMatrix();

        // Co-Driver Coralba / Monit Rally Computer (Center/Right Dash at x = +0.18f, y = 0.72f, z = 0.14f)
        glPushMatrix();
        glTranslatef(0.18f, 0.72f, 0.138f);
        glRotatef(180.0f, 0, 1, 0);
        glRotatef(10.0f, 0, 1, 0);
        glDisable(GL_LIGHTING);
        glColor3f(0.04f, 0.04f, 0.05f);
        glBegin(GL_QUADS);
        glNormal3f(0, 0, 1);
        glVertex3f(-0.075f, -0.042f, 0.0f);
        glVertex3f( 0.075f, -0.042f, 0.0f);
        glVertex3f( 0.075f,  0.042f, 0.0f);
        glVertex3f(-0.075f,  0.042f, 0.0f);
        glEnd();

        glPushMatrix();
        glTranslatef(0.0f, 0.015f, 0.002f);
        drawTextLocal("04:18.2", 0.0032f, 0.0052f, 0.98f, 0.15f, 0.15f, true);
        glPopMatrix();
        glPushMatrix();
        glTranslatef(0.0f, -0.018f, 0.002f);
        drawTextLocal("TRIP 14.8", 0.0028f, 0.0045f, 0.98f, 0.55f, 0.15f, true);
        glPopMatrix();
        glEnable(GL_LIGHTING);
        glPopMatrix();

        // Center Console: Sequential Shifter & WRC Fly-Off Hydraulic Handbrake
        glColor3f(0.10f, 0.10f, 0.12f);
        drawBox(-0.11f, 0.22f, -0.42f, 0.11f, 0.46f, 0.14f);
        drawPipe(Vec3(0.0f, 0.46f, -0.08f), Vec3(0.0f, 0.72f, -0.10f), 0.012f, 8);
        glColor3f(0.06f, 0.06f, 0.08f);
        drawBox(-0.024f, 0.71f, -0.124f, 0.024f, 0.76f, -0.076f);
        glColor3f(0.75f, 0.76f, 0.80f);
        drawPipe(Vec3(-0.045f, 0.46f, -0.02f), Vec3(-0.045f, 0.84f, -0.04f), 0.011f, 8);
        glColor3f(0.88f, 0.10f, 0.12f);
        drawPipe(Vec3(-0.045f, 0.68f, -0.034f), Vec3(-0.045f, 0.85f, -0.042f), 0.017f, 8);

        // Rear quarter window crew names & flags (flush against glass)
        struct CrewInfo { const char* d; const char* c; float flag[3][3]; };
        static const CrewInfo crews[3] = {
            {"S. OGIER",    "V. LANDAIS",   {{0.05f,0.22f,0.85f},{0.95f,0.95f,0.95f},{0.88f,0.08f,0.08f}}},
            {"T. NEUVILLE", "M. WYDAEGHE",  {{0.10f,0.10f,0.10f},{0.96f,0.85f,0.10f},{0.88f,0.08f,0.08f}}},
            {"A. FOURMAUX", "A. CORIA",     {{0.05f,0.22f,0.85f},{0.95f,0.95f,0.95f},{0.88f,0.08f,0.08f}}}
        };
        const CrewInfo& cr = crews[m % 3];

        for (float side : {-1.0f, 1.0f}) {
            glPushMatrix();
            glTranslatef(side * 0.732f, 1.05f, -0.70f);
            glRotatef(side > 0 ? 90.0f : -90.0f, 0, 1, 0);
            glRotatef(18.0f, 1, 0, 0); // flush with inward window slope
            glDisable(GL_LIGHTING);

            // Tinted backing strip on quarter glass (26cm x 11cm)
            glColor3f(0.04f, 0.05f, 0.07f);
            glBegin(GL_QUADS);
            glNormal3f(0, 0, 1);
            glVertex3f(-0.130f, -0.055f, 0.0f);
            glVertex3f( 0.130f, -0.055f, 0.0f);
            glVertex3f( 0.130f,  0.055f, 0.0f);
            glVertex3f(-0.130f,  0.055f, 0.0f);
            glEnd();

            // Driver row flag
            float fx = -0.115f, fy = 0.022f;
            for (int b = 0; b < 3; ++b) {
                glColor3fv(cr.flag[b]);
                glBegin(GL_QUADS);
                glVertex3f(fx + b * 0.009f, fy - 0.010f, 0.001f);
                glVertex3f(fx + (b + 1) * 0.009f, fy - 0.010f, 0.001f);
                glVertex3f(fx + (b + 1) * 0.009f, fy + 0.010f, 0.001f);
                glVertex3f(fx + b * 0.009f, fy + 0.010f, 0.001f);
                glEnd();
            }

            // Driver Name (crisp pure white)
            drawTextLocalAlign(cr.d, -0.078f, 0.022f, 0.0044f, 0.0070f, 1.0f, 1.0f, 1.0f, -1, true, 0.002f);

            // Co-driver row flag
            float cfy = -0.022f;
            for (int b = 0; b < 3; ++b) {
                glColor3fv(cr.flag[b]);
                glBegin(GL_QUADS);
                glVertex3f(fx + b * 0.009f, cfy - 0.010f, 0.001f);
                glVertex3f(fx + (b + 1) * 0.009f, cfy - 0.010f, 0.001f);
                glVertex3f(fx + (b + 1) * 0.009f, cfy + 0.010f, 0.001f);
                glVertex3f(fx + b * 0.009f, cfy + 0.010f, 0.001f);
                glEnd();
            }

            // Co-driver Name (crisp cyan/light-blue)
            drawTextLocalAlign(cr.c, -0.078f, -0.022f, 0.0040f, 0.0064f, 0.82f, 0.94f, 1.0f, -1, false, 0.002f);

            glEnable(GL_LIGHTING);
            glPopMatrix();
        }

        // Official FIA Competition Door Number Plates (Front Left & Right Doors)
        static const char* doorNum[3] = {"18", "11", "16"};
        for (float side : {-1.0f, 1.0f}) {
            glPushMatrix();
            glTranslatef(side * 0.928f, 0.64f, 0.10f);
            glRotatef(side > 0 ? 90.0f : -90.0f, 0, 1, 0);
            glDisable(GL_LIGHTING);

            // Black border frame (38cm x 34cm)
            glColor3f(0.04f, 0.04f, 0.05f);
            glBegin(GL_QUADS);
            glNormal3f(0, 0, 1);
            glVertex3f(-0.190f, -0.170f, 0.0f);
            glVertex3f( 0.190f, -0.170f, 0.0f);
            glVertex3f( 0.190f,  0.170f, 0.0f);
            glVertex3f(-0.190f,  0.170f, 0.0f);
            glEnd();

            // Brilliant white competition plate field
            glColor3f(0.98f, 0.98f, 0.99f);
            glBegin(GL_QUADS);
            glVertex3f(-0.180f, -0.160f, 0.001f);
            glVertex3f( 0.180f, -0.160f, 0.001f);
            glVertex3f( 0.180f,  0.160f, 0.001f);
            glVertex3f(-0.180f,  0.160f, 0.001f);
            glEnd();

            // Top FIA WRC header bar
            glColor3f(0.06f, 0.14f, 0.52f);
            glBegin(GL_QUADS);
            glVertex3f(-0.180f, 0.095f, 0.002f);
            glVertex3f( 0.180f, 0.095f, 0.002f);
            glVertex3f( 0.180f, 0.160f, 0.002f);
            glVertex3f(-0.180f, 0.160f, 0.002f);
            glEnd();

            glPushMatrix();
            glTranslatef(0.0f, 0.128f, 0.003f);
            drawTextLocal("FIA WRC", 0.0040f, 0.0068f, 0.98f, 0.98f, 0.98f, true);
            glPopMatrix();

            // Large Bold Competition Number
            glPushMatrix();
            glTranslatef(0.0f, -0.015f, 0.003f);
            drawTextLocal(doorNum[m % 3], 0.017f, 0.024f, 0.05f, 0.05f, 0.08f, true);
            glPopMatrix();

            // Bottom Rally1 Hybrid strip
            glColor3f(0.85f, 0.08f, 0.08f);
            glBegin(GL_QUADS);
            glVertex3f(-0.180f, -0.160f, 0.002f);
            glVertex3f( 0.180f, -0.160f, 0.002f);
            glVertex3f( 0.180f, -0.130f, 0.002f);
            glVertex3f(-0.180f, -0.130f, 0.002f);
            glEnd();

            glPushMatrix();
            glTranslatef(0.0f, -0.145f, 0.003f);
            drawTextLocal("RALLY1 HYBRID", 0.0028f, 0.0040f, 0.98f, 0.98f, 0.98f, true);
            glPopMatrix();

            glEnable(GL_LIGHTING);
            glPopMatrix();
        }

        const float noSpec[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, noSpec);

        glEndList();
    }

    // ---- stationary brake caliper & suspension knuckle (does not spin with wheel)
    if (g_listBrakeStationary) glDeleteLists(g_listBrakeStationary, 1);
    g_listBrakeStationary = glGenLists(1);
    glNewList(g_listBrakeStationary, GL_COMPILE);
    // Suspension knuckle / upright hub mount
    glColor3f(0.22f, 0.22f, 0.25f);
    drawBox(-0.02f, -0.06f, -0.08f, 0.045f, 0.16f, 0.08f);
    // Red Brembo 4-piston monobloc caliper at upper front of rotor
    glColor3f(0.88f, 0.06f, 0.06f);
    drawBox(0.032f, 0.105f, -0.075f, 0.095f, 0.225f, 0.075f);
    // Caliper bridge & curved twin piston wells
    glColor3f(0.72f, 0.05f, 0.05f);
    drawBox(0.040f, 0.130f, -0.082f, 0.090f, 0.200f, 0.082f);
    // White Brembo racing logo stripe along outer face
    glDisable(GL_LIGHTING);
    glColor3f(0.98f, 0.98f, 1.0f);
    glBegin(GL_QUADS);
    glNormal3f(1, 0, 0);
    glVertex3f(0.096f, 0.155f, -0.055f);
    glVertex3f(0.096f, 0.155f,  0.055f);
    glVertex3f(0.096f, 0.175f,  0.055f);
    glVertex3f(0.096f, 0.175f, -0.055f);
    glEnd();
    // Silver pad retention pins & spring clips
    glColor3f(0.82f, 0.82f, 0.85f);
    drawBox(0.035f, 0.195f, -0.040f, 0.092f, 0.205f, -0.030f);
    drawBox(0.035f, 0.195f,  0.030f, 0.092f, 0.205f,  0.040f);
    glEnable(GL_LIGHTING);
    glEndList();

    // ---- rotating wheel assembly (spins around X axis, outer face = +X)
    if (g_listWheel) glDeleteLists(g_listWheel, 1);
    g_listWheel = glGenLists(1);
    glNewList(g_listWheel, GL_COMPILE);
    const float w2 = 0.145f;
    const int segs = 32;

    // 1. Tire Tread (textured)
    texOn(g_texTire);
    glColor3f(1.0f, 1.0f, 1.0f);
    glBegin(GL_QUAD_STRIP);
    for (int i = 0; i <= segs; ++i) {
        float a = 2.0f * PI * i / segs, cy = std::cos(a), cz = std::sin(a);
        glNormal3f(0, cy, cz);
        glTexCoord2f((float)i / segs * 2.0f, 0.0f); glVertex3f(-w2, cy * WHEEL_R, cz * WHEEL_R);
        glTexCoord2f((float)i / segs * 2.0f, 1.0f); glVertex3f(w2, cy * WHEEL_R, cz * WHEEL_R);
    }
    glEnd();
    texOff();

    // 2. Matte Rubber Sidewalls
    glColor3f(0.07f, 0.07f, 0.08f);
    for (int side = -1; side <= 1; side += 2) {
        glBegin(GL_QUAD_STRIP);
        glNormal3f((float)side, 0, 0);
        for (int i = 0; i <= segs; ++i) {
            float a = 2.0f * PI * i / segs, cy = std::cos(a), cz = std::sin(a);
            glVertex3f(side * (w2 + 0.004f), cy * WHEEL_R * 0.985f, cz * WHEEL_R * 0.985f);
            glVertex3f(side * (w2 + 0.012f), cy * 0.225f, cz * 0.225f);
        }
        glEnd();
    }

    // 3. Yellow/White Pirelli / Michelin Tire Sidewall Stencil Decals (Outer face +X)
    glDisable(GL_LIGHTING);
    glBegin(GL_QUADS);
    glNormal3f(1, 0, 0);
    for (float baseA : { PI * 0.40f, PI * 1.40f }) {
        glColor3f(0.96f, 0.84f, 0.12f); // Vibrant rally yellow
        for (int k = 0; k < 6; ++k) {
            float a0 = baseA + k * 0.045f, a1 = a0 + 0.038f;
            float r0 = 0.260f, r1 = 0.285f;
            glVertex3f(w2 + 0.009f, std::cos(a0) * r0, std::sin(a0) * r0);
            glVertex3f(w2 + 0.009f, std::cos(a1) * r0, std::sin(a1) * r0);
            glVertex3f(w2 + 0.009f, std::cos(a1) * r1, std::sin(a1) * r1);
            glVertex3f(w2 + 0.009f, std::cos(a0) * r1, std::sin(a0) * r1);
        }
        glColor3f(0.95f, 0.95f, 0.98f); // White accent marking
        float aW = baseA + 0.29f;
        glVertex3f(w2 + 0.009f, std::cos(aW) * 0.262f, std::sin(aW) * 0.262f);
        glVertex3f(w2 + 0.009f, std::cos(aW + 0.04f) * 0.262f, std::sin(aW + 0.04f) * 0.262f);
        glVertex3f(w2 + 0.009f, std::cos(aW + 0.04f) * 0.282f, std::sin(aW + 0.04f) * 0.282f);
        glVertex3f(w2 + 0.009f, std::cos(aW) * 0.282f, std::sin(aW) * 0.282f);
    }
    glEnd();
    glEnable(GL_LIGHTING);

    // 4. Ventilated Cross-Drilled Brake Rotor & Bell
    // Rotor friction swept ring (cast iron metallic)
    glColor3f(0.38f, 0.39f, 0.41f);
    glBegin(GL_QUAD_STRIP);
    glNormal3f(1, 0, 0);
    for (int i = 0; i <= segs; ++i) {
        float a = 2.0f * PI * i / segs, cy = std::cos(a), cz = std::sin(a);
        glVertex3f(0.055f, cy * 0.125f, cz * 0.125f);
        glVertex3f(0.055f, cy * 0.210f, cz * 0.210f);
    }
    glEnd();
    // Inner aluminum bell/hat (hard anodized dark charcoal)
    glColor3f(0.16f, 0.17f, 0.19f);
    glBegin(GL_TRIANGLE_FAN);
    glNormal3f(1, 0, 0); glVertex3f(0.053f, 0, 0);
    for (int i = 0; i <= segs; ++i) { float a = 2.0f * PI * i / segs; glVertex3f(0.053f, std::cos(a) * 0.125f, std::sin(a) * 0.125f); }
    glEnd();
    // Cross-drilled cooling hole pattern on the brake rotor surface
    glDisable(GL_LIGHTING);
    glColor3f(0.12f, 0.12f, 0.14f);
    glBegin(GL_POINTS);
    glPointSize(2.0f);
    for (int h = 0; h < 16; ++h) {
        float a = 2.0f * PI * h / 16;
        for (float r : {0.145f, 0.168f, 0.192f}) {
            glVertex3f(0.056f, std::cos(a + r * 1.5f) * r, std::sin(a + r * 1.5f) * r);
        }
    }
    glEnd();
    // 10 Silver Rotor Drive Bobbins / Mounting Bolts
    glColor3f(0.78f, 0.78f, 0.82f);
    glBegin(GL_POINTS);
    glPointSize(3.0f);
    for (int b = 0; b < 10; ++b) {
        float a = 2.0f * PI * b / 10;
        glVertex3f(0.056f, std::cos(a) * 0.125f, std::sin(a) * 0.125f);
    }
    glEnd();
    glPointSize(1.0f);
    glEnable(GL_LIGHTING);

    // 5. Outer Alloy Rim Barrel & Stepped Lip
    glColor3f(0.86f, 0.87f, 0.90f);
    glBegin(GL_QUAD_STRIP);
    glNormal3f(1, 0, 0);
    for (int i = 0; i <= segs; ++i) {
        float a = 2.0f * PI * i / segs, cy = std::cos(a), cz = std::sin(a);
        glVertex3f(w2 + 0.015f, cy * 0.228f, cz * 0.228f);
        glVertex3f(w2 + 0.015f, cy * 0.204f, cz * 0.204f);
    }
    glEnd();
    // Inner rim barrel well
    glColor3f(0.48f, 0.49f, 0.52f);
    glBegin(GL_QUAD_STRIP);
    for (int i = 0; i <= segs; ++i) {
        float a = 2.0f * PI * i / segs, cy = std::cos(a), cz = std::sin(a);
        glNormal3f(0, -cy, -cz);
        glVertex3f(w2 + 0.015f, cy * 0.204f, cz * 0.204f);
        glVertex3f(0.060f,     cy * 0.204f, cz * 0.204f);
    }
    glEnd();

    // 6. 10-Spoke Rally Wheel (OZ Racing / Speedline style)
    glColor3f(0.88f, 0.88f, 0.91f);
    for (int sp = 0; sp < 10; ++sp) {
        float a = 2.0f * PI * sp / 10.0f, ca = std::cos(a), sa = std::sin(a);
        float pa = a + PI * 0.5f, cp2 = std::cos(pa) * 0.013f, sp2 = std::sin(pa) * 0.013f;
        glBegin(GL_QUADS);
        glNormal3f(1, 0, 0);
        glVertex3f(w2 + 0.012f, ca * 0.045f - cp2, sa * 0.045f - sp2);
        glVertex3f(w2 + 0.014f, ca * 0.204f - cp2 * 1.15f, sa * 0.204f - sp2 * 1.15f);
        glVertex3f(w2 + 0.014f, ca * 0.204f + cp2 * 1.15f, sa * 0.204f + sp2 * 1.15f);
        glVertex3f(w2 + 0.012f, ca * 0.045f + cp2, sa * 0.045f + sp2);
        glEnd();
    }

    // 7. Recessed Center Bowl & Anodized Center-Lock Nut
    glColor3f(0.14f, 0.14f, 0.16f);
    glBegin(GL_TRIANGLE_FAN);
    glNormal3f(1, 0, 0); glVertex3f(w2 + 0.008f, 0, 0);
    for (int i = 0; i <= 16; ++i) { float a = 2.0f * PI * i / 16; glVertex3f(w2 + 0.008f, std::cos(a) * 0.046f, std::sin(a) * 0.046f); }
    glEnd();

    // Anodized Red Center-Lock Hex Nut
    glColor3f(0.88f, 0.12f, 0.12f);
    glBegin(GL_QUAD_STRIP);
    for (int i = 0; i <= 6; ++i) {
        float a = 2.0f * PI * i / 6.0f, cy = std::cos(a), cz = std::sin(a);
        glNormal3f(0, cy, cz);
        glVertex3f(w2 + 0.008f, cy * 0.026f, cz * 0.026f);
        glVertex3f(w2 + 0.028f, cy * 0.024f, cz * 0.024f);
    }
    glEnd();
    glBegin(GL_TRIANGLE_FAN);
    glNormal3f(1, 0, 0); glVertex3f(w2 + 0.028f, 0, 0);
    for (int i = 0; i <= 6; ++i) { float a = 2.0f * PI * i / 6.0f; glVertex3f(w2 + 0.028f, std::cos(a) * 0.024f, std::sin(a) * 0.024f); }
    glEnd();

    // Center safety locking pin
    glColor3f(0.92f, 0.92f, 0.95f);
    drawBox(w2 + 0.026f, -0.006f, -0.018f, w2 + 0.030f, 0.006f, 0.018f);

    glEndList();
}


static void drawCar(const RallyCar& c) {
    glPushMatrix();
    glTranslatef(c.pos.x, c.pos.y, c.pos.z);
    glRotatef(c.heading * RAD2DEG, 0, 1, 0);
    glRotatef(c.attPitch * RAD2DEG, -1, 0, 0);
    glRotatef(c.attRoll * RAD2DEG, 0, 0, 1);

    // 4 wheels coordinates
    const float wx[4] = { WHEEL_X, -WHEEL_X,  WHEEL_X, -WHEEL_X};
    const float wz[4] = { AXLE_F,   AXLE_F,   AXLE_R,   AXLE_R};

    // Soft ground contact shadow under chassis and under 4 tires
    if (!c.airborne) {
        glDisable(GL_LIGHTING);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        texOn(g_texShadow);
        glColor4f(0, 0, 0, 0.78f);
        glBegin(GL_QUADS);
        glNormal3f(0, 1, 0);
        glTexCoord2f(0, 0); glVertex3f(-1.25f, 0.03f, -2.35f);
        glTexCoord2f(1, 0); glVertex3f( 1.25f, 0.03f, -2.35f);
        glTexCoord2f(1, 1); glVertex3f( 1.25f, 0.03f,  2.35f);
        glTexCoord2f(0, 1); glVertex3f(-1.25f, 0.03f,  2.35f);

        // Darker contact patch shadows under tires
        glColor4f(0, 0, 0, 0.88f);
        for (int w = 0; w < 4; ++w) {
            float tx0 = wx[w] - 0.24f, tx1 = wx[w] + 0.24f;
            float tz0 = wz[w] - 0.40f, tz1 = wz[w] + 0.40f;
            glTexCoord2f(0, 0); glVertex3f(tx0, 0.035f, tz0);
            glTexCoord2f(1, 0); glVertex3f(tx1, 0.035f, tz0);
            glTexCoord2f(1, 1); glVertex3f(tx1, 0.035f, tz1);
            glTexCoord2f(0, 1); glVertex3f(tx0, 0.035f, tz1);
        }
        glEnd();
        texOff();
        glDisable(GL_BLEND);
        glEnable(GL_LIGHTING);
    }

    // 4 wheels: FL, FR, RL, RR
    for (int i = 0; i < 4; ++i) {
        glPushMatrix();
        glTranslatef(wx[i], WHEEL_R, wz[i]);
        if (i < 2) glRotatef(c.steerAngle * RAD2DEG, 0, 1, 0);
        if (wx[i] < 0) glRotatef(180.0f, 0, 1, 0);

        // 1. Stationary suspension knuckle & Brembo 4-piston caliper
        if (g_listBrakeStationary) glCallList(g_listBrakeStationary);

        // 2. Dynamic glowing cherry-red brake rotor ring under hard braking
        if (c.brakeHeat > 0.04f) {
            glDisable(GL_LIGHTING);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE);
            float h = c.brakeHeat;
            glColor4f(1.0f, 0.22f + 0.35f * (1.0f - h), 0.04f, clampf(h * 0.92f, 0.0f, 0.90f));
            glBegin(GL_QUAD_STRIP);
            for (int k = 0; k <= 24; ++k) {
                float a = 2.0f * PI * k / 24;
                float ca = std::cos(a), sa = std::sin(a);
                glVertex3f(0.057f, ca * 0.125f, sa * 0.125f);
                glVertex3f(0.057f, ca * 0.210f, sa * 0.210f);
            }
            glEnd();
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glDisable(GL_BLEND);
            glEnable(GL_LIGHTING);
        }

        // 3. Rotating wheel rim, spokes, sidewall markings & rotor disc
        float spinSign = (wx[i] < 0) ? -1.0f : 1.0f;
        glRotatef(c.wheelRot * RAD2DEG * spinSign, 1, 0, 0);
        glCallList(g_listWheel);
        glPopMatrix();
    }

    // Sprung chassis: apply suspension bounce, body roll & body pitch
    glTranslatef(0.0f, c.suspBounce, 0.0f);
    glRotatef(c.bodyRoll * RAD2DEG, 0, 0, 1);
    glRotatef(c.bodyPitch * RAD2DEG, -1, 0, 0);
    glCallList(g_listCarBody[c.model % 3]);

    // Flexible Heavy-Duty Polyurethane Rally Mudflaps (behind all 4 wheels)
    float flutter = clampf(c.vx * 0.0035f, -0.04f, 0.12f);
    for (float side : {-1.0f, 1.0f}) {
        // Front mudflap
        glPushMatrix();
        glTranslatef(side * 0.91f, 0.36f, AXLE_F - 0.40f);
        glRotatef(-flutter * RAD2DEG, 1, 0, 0);
        glColor3f(0.85f, 0.08f, 0.08f); // Rally red
        drawBox(-0.13f, -0.32f, -0.012f, 0.13f, 0.0f, 0.012f);
        glColor3f(0.72f, 0.74f, 0.78f); // Top mounting bracket
        drawBox(-0.135f, -0.04f, -0.016f, 0.135f, 0.015f, 0.016f);
        glPopMatrix();

        // Rear mudflap
        glPushMatrix();
        glTranslatef(side * 0.91f, 0.36f, AXLE_R - 0.40f);
        glRotatef(-flutter * RAD2DEG, 1, 0, 0);
        glColor3f(0.85f, 0.08f, 0.08f);
        drawBox(-0.13f, -0.32f, -0.012f, 0.13f, 0.0f, 0.012f);
        glColor3f(0.72f, 0.74f, 0.78f);
        drawBox(-0.135f, -0.04f, -0.016f, 0.135f, 0.015f, 0.016f);
        glPopMatrix();
    }

    // Articulated windshield wiper
    glPushMatrix();
    glTranslatef(0.18f, 0.72f, 0.36f);
    glRotatef(25.0f, 1, 0, 0); // windscreen rake angle
    glRotatef(c.wiperAngle * 42.0f, 0, 0, 1);
    glColor3f(0.10f, 0.10f, 0.12f);
    drawBox(-0.012f, 0.0f, -0.008f, 0.012f, 0.38f, 0.008f); // arm
    drawBox(-0.006f, 0.18f, -0.012f, 0.006f, 0.48f, 0.012f); // blade
    glPopMatrix();

    // Volumetric 3D Headlight Beams in fog, sunset, or rain
    bool lampsOn = (g_climate == 1 || g_climate == 3);
    if (lampsOn) {
        glDisable(GL_LIGHTING);
        glDepthMask(GL_FALSE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        for (float hx : {-0.72f, 0.72f}) {
            glBegin(GL_TRIANGLE_FAN);
            glColor4f(1.0f, 0.98f, 0.86f, 0.18f);
            glVertex3f(hx, 0.52f, 1.85f); // headlight source at 3D LED cluster
            glColor4f(0.95f, 0.95f, 1.0f, 0.0f); // fade out at 38m
            for (int k = 0; k <= 14; ++k) {
                float a = 2.0f * PI * k / 14;
                glVertex3f(hx * 1.8f + std::cos(a) * 2.2f, 0.25f + std::sin(a) * 1.6f, 38.0f);
            }
            glEnd();
        }
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDisable(GL_BLEND);
        glDepthMask(GL_TRUE);
        glEnable(GL_LIGHTING);
    }

    auto zCurveRear = [](float x) -> float {
        float f = x / 0.94f;
        return -2.06f + 0.18f * (f * f);
    };

    // Dynamic brake lights & blooming coronas
    glDisable(GL_LIGHTING);
    if (c.brake > 0.05f || c.handbrake > 0.1f) {
        glColor3f(1.0f, 0.08f, 0.06f);
        for (float side : {-1.0f, 1.0f}) {
            glBegin(GL_QUAD_STRIP);
            for (int k = 0; k <= 6; ++k) {
                float x = side * lerpf(0.48f, 0.88f, (float)k / 6.0f);
                float z = zCurveRear(x) - 0.030f;
                glVertex3f(x, 0.63f, z);
                glVertex3f(x, 0.74f, z);
            }
            glEnd();
        }
        // Center full-width connecting brake strip
        glBegin(GL_QUAD_STRIP);
        for (int k = 0; k <= 8; ++k) {
            float x = lerpf(-0.48f, 0.48f, (float)k / 8.0f);
            float z = zCurveRear(x) - 0.030f;
            glVertex3f(x, 0.675f, z);
            glVertex3f(x, 0.705f, z);
        }
        glEnd();

        // High-mount FIA rain/brake light
        glBegin(GL_QUADS);
        glVertex3f(-0.14f, 1.32f, -1.95f);
        glVertex3f( 0.14f, 1.32f, -1.95f);
        glVertex3f( 0.14f, 1.38f, -1.95f);
        glVertex3f(-0.14f, 1.38f, -1.95f);
        glEnd();

        // Soft blooming corona billboard flare
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        texOn(g_texSoft);
        for (float tx : {-0.67f, 0.67f}) {
            float tz = zCurveRear(tx) - 0.04f;
            glColor4f(1.0f, 0.15f, 0.08f, 0.65f);
            glBegin(GL_QUADS);
            glTexCoord2f(0, 0); glVertex3f(tx - 0.28f, 0.68f - 0.28f, tz);
            glTexCoord2f(1, 0); glVertex3f(tx + 0.28f, 0.68f - 0.28f, tz);
            glTexCoord2f(1, 1); glVertex3f(tx + 0.28f, 0.68f + 0.28f, tz);
            glTexCoord2f(0, 1); glVertex3f(tx - 0.28f, 0.68f + 0.28f, tz);
            glEnd();
        }
        texOff();
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDisable(GL_BLEND);
    }
    // Reverse LED lights
    if (c.gear == 0) {
        glColor3f(0.95f, 0.98f, 1.0f);
        for (float rx : {-0.31f, 0.31f}) {
            float rz = zCurveRear(rx) - 0.025f;
            glBegin(GL_QUADS);
            glVertex3f(rx - 0.05f, 0.33f, rz);
            glVertex3f(rx + 0.05f, 0.33f, rz);
            glVertex3f(rx + 0.05f, 0.37f, rz);
            glVertex3f(rx - 0.05f, 0.37f, rz);
            glEnd();
        }
    }

    // Dynamic Anti-Lag (ALS) / Hybrid Boost twin exhaust flames
    if (c.hybridActive || c.backfireTimer > 0.0f) {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        float baseLen = c.hybridActive ? (0.35f + 0.18f * std::sin(g_timeAccum * 55.0f))
                                       : (0.42f * c.backfireIntensity + randRange(-0.06f, 0.06f));
        for (float ex : {-0.42f, 0.42f}) {
            float ez = zCurveRear(ex) - 0.02f;
            float ey = 0.24f;
            float tipZ = ez - baseLen;

            // Outer turbulent flame plume cone (crossed quads with gradient)
            for (int rot = 0; rot < 3; ++rot) {
                float a = (float)rot * (PI / 3.0f);
                float dx = std::cos(a) * 0.065f, dy = std::sin(a) * 0.065f;
                glBegin(GL_TRIANGLES);
                if (c.hybridActive) glColor4f(0.15f, 0.78f, 1.0f, 0.85f);
                else                glColor4f(1.0f, 0.36f, 0.06f, 0.85f);
                glVertex3f(ex + dx, ey + dy, ez);
                glVertex3f(ex - dx, ey - dy, ez);
                glColor4f(1.0f, 0.15f, 0.02f, 0.0f);
                glVertex3f(ex, ey, tipZ);
                glEnd();
            }

            // Inner bright incandescent supersonic core
            for (int rot = 0; rot < 3; ++rot) {
                float a = (float)rot * (PI / 3.0f) + PI * 0.16f;
                float dx = std::cos(a) * 0.038f, dy = std::sin(a) * 0.038f;
                glBegin(GL_TRIANGLES);
                if (c.hybridActive) glColor4f(0.85f, 0.95f, 1.0f, 0.95f);
                else                glColor4f(1.0f, 0.95f, 0.65f, 0.95f);
                glVertex3f(ex + dx, ey + dy, ez);
                glVertex3f(ex - dx, ey - dy, ez);
                glColor4f(1.0f, 0.65f, 0.15f, 0.0f);
                glVertex3f(ex, ey, ez - baseLen * 0.55f);
                glEnd();
            }
        }
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDisable(GL_BLEND);
    }
    glEnable(GL_LIGHTING);
    glPopMatrix();
}

// ============================================================================
//  LINUX /dev/input/js0 JOYSTICK & STEERING WHEEL SUPPORT
// ============================================================================
static void initJoystick() {
    g_joy = JoyState();
    if (const char* s = std::getenv("WRC_AXIS_STEER"))    g_joy.axisSteer = std::atoi(s);
    if (const char* s = std::getenv("WRC_AXIS_THROTTLE")) g_joy.axisThrottle = std::atoi(s);
    if (const char* s = std::getenv("WRC_AXIS_BRAKE"))    g_joy.axisBrake = std::atoi(s);
    g_joy.fd = open("/dev/input/js0", O_RDONLY | O_NONBLOCK);
}

static void pollJoystick() {
    if (g_joy.fd < 0) {
        if (++g_joy.retry > 180) {
            g_joy.retry = 0;
            g_joy.fd = open("/dev/input/js0", O_RDONLY | O_NONBLOCK);
        }
        return;
    }
    struct js_event ev;
    while (read(g_joy.fd, &ev, sizeof(ev)) == sizeof(ev)) {
        bool isInit = (ev.type & JS_EVENT_INIT) != 0;
        uint8_t type = ev.type & ~JS_EVENT_INIT;
        if (type == JS_EVENT_AXIS && ev.number < 16) {
            float v = clampf(ev.value / 32767.0f, -1.0f, 1.0f);
            g_joy.axis[ev.number] = v;
            if (isInit || !g_joy.haveRest[ev.number]) {
                g_joy.rest[ev.number] = v;
                g_joy.haveRest[ev.number] = true;
            } else if (std::fabs(v - g_joy.rest[ev.number]) > 0.08f) {
                g_joy.active = true;
            }
        } else if (type == JS_EVENT_BUTTON && !isInit) {
            g_joy.active = true;
            if (ev.number == 0 || ev.number == 1) g_joy.handbrake = (ev.value != 0);
            else if (ev.number == 2 && ev.value) g_cameraMode = (g_cameraMode + 1) % 4;
            else if (ev.number == 3 && ev.value) g_car.model = (g_car.model + 1) % 3;
        }
    }
    if (errno != EAGAIN && errno != EWOULDBLOCK && errno != 0) {
        close(g_joy.fd);
        g_joy.fd = -1;
        g_joy.active = false;
        return;
    }
    if (!g_joy.active) {
        g_joy.steer = g_joy.throttle = g_joy.brake = 0.0f;
        return;
    }
    float st = -g_joy.axis[g_joy.axisSteer];
    g_joy.steer = (std::fabs(st) < 0.06f) ? 0.0f : st;

    auto readPedal = [&](int ax) -> float {
        if (ax < 0 || ax >= 16 || !g_joy.haveRest[ax]) return 0.0f;
        float r = g_joy.rest[ax], v = g_joy.axis[ax];
        if (r < -0.5f) return clampf((v + 1.0f) * 0.5f, 0.0f, 1.0f); // resting at -1
        return clampf(std::fabs(v - r), 0.0f, 1.0f);
    };
    if (g_joy.axisThrottle >= 0) g_joy.throttle = readPedal(g_joy.axisThrottle);
    else g_joy.throttle = std::max(clampf(-g_joy.axis[1], 0.0f, 1.0f), readPedal(5));

    if (g_joy.axisBrake >= 0) g_joy.brake = readPedal(g_joy.axisBrake);
    else g_joy.brake = std::max(clampf(g_joy.axis[1], 0.0f, 1.0f), readPedal(2));
}

// ============================================================================
//  PARTICLES, SKIDMARKS & WRC RALLY1 HYBRID VEHICLE PHYSICS
// ============================================================================
static void pushSkid(const Vec3& a, const Vec3& b, float alpha) {
    if (g_skids.size() < MAX_SKIDS) {
        g_skids.push_back({a, b, alpha});
    } else {
        g_skids[g_skidHead] = {a, b, alpha};
        g_skidHead = (g_skidHead + 1) % MAX_SKIDS;
    }
}

static void spawnDust(const Vec3& pos, const Vec3& vel, float scale) {
    if (g_fx.size() > 650) return;
    const ClimateProfile& cp = CLIMATES[g_climate];
    Particle3D p{};
    p.pos = pos + Vec3(randRange(-0.22f, 0.22f), 0.10f, randRange(-0.22f, 0.22f));
    p.vel = vel * 0.25f + Vec3(randRange(-2.4f, 2.4f), randRange(1.0f, 3.2f), randRange(-2.4f, 2.4f));
    p.color[0] = cp.dust[0];
    p.color[1] = cp.dust[1];
    p.color[2] = cp.dust[2];
    p.color[3] = clampf(0.42f * scale, 0.15f, 0.65f);
    p.size = 0.35f + 0.35f * scale;
    p.growth = 1.6f;
    p.maxLife = randRange(0.45f, 0.85f);
    p.life = p.maxLife;
    p.kind = 1;
    g_fx.push_back(p);
}

static void spawnSparks(const Vec3& pos, const Vec3& normal) {
    for (int i = 0; i < 14 && g_fx.size() < 750; ++i) {
        Particle3D p{};
        p.pos = pos + Vec3(0, 0.5f, 0);
        p.vel = normal * randRange(2.0f, 6.0f) + Vec3(randRange(-4.0f, 4.0f), randRange(1.5f, 5.5f), randRange(-4.0f, 4.0f));
        p.color[0] = 1.0f; p.color[1] = randRange(0.55f, 0.92f); p.color[2] = 0.15f; p.color[3] = 1.0f;
        p.size = 0.12f;
        p.growth = 0.0f;
        p.maxLife = randRange(0.18f, 0.42f);
        p.life = p.maxLife;
        p.kind = 2;
        g_fx.push_back(p);
    }
}

static void initWeather() {
    g_weather.clear();
    g_fx.clear();
    g_skids.clear();
    g_skidHead = 0;
    g_weather.resize(900);
    for (Particle3D& p : g_weather) {
        p.pos = g_car.pos + Vec3(randRange(-55.0f, 55.0f), randRange(0.0f, 34.0f), randRange(-55.0f, 55.0f));
        p.vel = Vec3(0, -16.0f, 0);
        p.size = 1.0f;
        p.life = rand01();
        p.maxLife = 1.0f;
        p.kind = 0;
    }
}

static void resetCarToNode(int idx) {
    if (g_nodes.empty()) return;
    int N = (int)g_nodes.size();
    idx = std::max(0, std::min(N - 1, idx));
    const StageNode& nd = g_nodes[idx];
    int prevModel = g_car.model;
    g_car = RallyCar();
    g_car.model = prevModel;
    g_car.pos = nd.pos;
    g_car.prevGroundY = nd.pos.y;
    g_car.heading = std::atan2(nd.forward.x, nd.forward.z);
    g_car.nearestNode = idx;
    g_prevNodeIdx = idx;

    Vec3 fwd(std::sin(g_car.heading), 0, std::cos(g_car.heading));
    g_camYaw = g_car.heading;
    g_camPos = g_car.pos - fwd * 7.2f + Vec3(0, 2.8f, 0);
    g_camTarget = g_car.pos + fwd * 4.5f + Vec3(0, 1.0f, 0);
}

static void resetStageSession() {
    int startIdx = std::min(5, (int)g_nodes.size() - 1);
    resetCarToNode(startIdx);
    g_countdown = 0.0f;
    g_goFlash = 0.0f;
    g_finished = false;
    g_finishTime = 0.0f;
    g_lap = 1;
    g_lapStart = 0.0f;
    initWeather();
}

static void resolveSceneryCollisions(RallyCar& c, const Vec3& fwd, const Vec3& left) {
    if (g_solidGrid.empty()) return;
    int cx = gridCell(c.pos.x, SOLID_CELL, SOLID_GRID_N);
    int cz = gridCell(c.pos.z, SOLID_CELL, SOLID_GRID_N);
    const float carR = 1.08f;

    for (int gz = std::max(0, cz - 1); gz <= std::min(SOLID_GRID_N - 1, cz + 1); ++gz)
        for (int gx = std::max(0, cx - 1); gx <= std::min(SOLID_GRID_N - 1, cx + 1); ++gx)
            for (int idx : g_solidGrid[(size_t)gz * SOLID_GRID_N + gx]) {
                const SceneryObject& o = g_scenery[idx];
                if (o.solid == 1) {
                    float r = o.kind == SCN_CONIFER ? 0.42f * o.scale :
                              (o.kind == SCN_BROADLEAF ? 0.48f * o.scale : o.radius * 0.72f);
                    float dx = c.pos.x - o.pos.x, dz = c.pos.z - o.pos.z;
                    float d2 = dx * dx + dz * dz, minD = carR + r;
                    if (d2 < minD * minD && d2 > 1e-6f) {
                        float d = std::sqrt(d2);
                        Vec3 n(dx / d, 0, dz / d);
                        c.pos += n * (minD - d + 0.02f);
                        Vec3 vWorld = fwd * c.vx + left * c.vy;
                        float vn = dot(vWorld, n);
                        if (vn < 0.0f) {
                            vWorld -= n * (1.35f * vn);
                            vWorld = vWorld * 0.72f;
                            c.vx = dot(vWorld, fwd);
                            c.vy = dot(vWorld, left);
                            c.yawRate += randRange(-1.2f, 1.2f);
                            g_shake = std::min(1.0f, g_shake + 0.45f);
                            if (c.hitCooldown <= 0.0f) {
                                spawnSparks(c.pos - n * carR, n);
                                c.hitCooldown = 0.20f;
                            }
                        }
                    }
                } else if (o.solid == 2) {
                    float rad = o.yaw * (PI / 180.0f);
                    float ca = std::cos(rad), sa = std::sin(rad);
                    float rx = c.pos.x - o.pos.x, rz = c.pos.z - o.pos.z;
                    float lx =  rx * ca - rz * sa;
                    float lz =  rx * sa + rz * ca;
                    float hx = o.w * 0.5f + carR, hz = o.d * 0.5f + carR;
                    if (std::fabs(lx) < hx && std::fabs(lz) < hz) {
                        float px = hx - std::fabs(lx), pz = hz - std::fabs(lz);
                        Vec3 nLocal = (px < pz) ? Vec3(sgnf(lx), 0, 0) : Vec3(0, 0, sgnf(lz));
                        float pen = std::min(px, pz) + 0.02f;
                        Vec3 n(nLocal.x * ca + nLocal.z * sa, 0, -nLocal.x * sa + nLocal.z * ca);
                        c.pos += n * pen;
                        Vec3 vWorld = fwd * c.vx + left * c.vy;
                        float vn = dot(vWorld, n);
                        if (vn < 0.0f) {
                            vWorld -= n * (1.30f * vn);
                            vWorld = vWorld * 0.68f;
                            c.vx = dot(vWorld, fwd);
                            c.vy = dot(vWorld, left);
                            g_shake = std::min(1.0f, g_shake + 0.55f);
                            if (c.hitCooldown <= 0.0f) {
                                spawnSparks(c.pos - n * carR, n);
                                c.hitCooldown = 0.20f;
                            }
                        }
                    }
                }
            }
}

static void stepPhysics(float dt) {
    if (g_nodes.size() < 4) return;
    int N = (int)g_nodes.size();
    RallyCar& c = g_car;
    if (c.hitCooldown > 0.0f) c.hitCooldown = std::max(0.0f, c.hitCooldown - dt);

    // 1. Gather controls (Keyboard / Joystick / Autopilot Demo)
    float rawThr = (g_keys[XK_w] || g_keys[XK_W] || g_keys[XK_Up]) ? 1.0f : 0.0f;
    float rawBrk = (g_keys[XK_s] || g_keys[XK_S] || g_keys[XK_Down]) ? 1.0f : 0.0f;
    float rawStr = 0.0f;
    if (g_keys[XK_a] || g_keys[XK_A] || g_keys[XK_Left])  rawStr += 1.0f;
    if (g_keys[XK_d] || g_keys[XK_D] || g_keys[XK_Right]) rawStr -= 1.0f;
    float rawHb  = g_keys[XK_space] ? 1.0f : 0.0f;

    if (g_joy.active) {
        rawThr = std::max(rawThr, g_joy.throttle);
        rawBrk = std::max(rawBrk, g_joy.brake);
        if (std::fabs(rawStr) < 0.01f) rawStr = g_joy.steer;
        if (g_joy.handbrake) rawHb = 1.0f;
    }

    if (g_autopilot && g_countdown <= 0.0f && !g_finished) {
        int look = (c.nearestNode + 7) % N;
        if (!g_stageClosed) look = std::min(N - 1, c.nearestNode + 7);
        Vec3 toTarget = g_nodes[look].pos - c.pos;
        float targetYaw = std::atan2(toTarget.x, toTarget.z);
        float err = wrapPi(targetYaw - c.heading);
        rawStr = clampf(err * 2.2f, -1.0f, 1.0f);
        float maxCurv = 0.0f;
        for (int k = 2; k <= 16; ++k) {
            int j = g_stageClosed ? (c.nearestNode + k) % N : std::min(N - 1, c.nearestNode + k);
            maxCurv = std::max(maxCurv, std::fabs(g_nodes[j].curvature));
        }
        float targetSpd = clampf(48.0f - maxCurv * 680.0f, 16.0f, 48.0f);
        rawThr = (c.vx < targetSpd) ? 0.92f : 0.15f;
        rawBrk = (c.vx > targetSpd + 5.0f) ? 0.75f : 0.0f;
        rawHb  = (maxCurv > 0.038f && c.vx > 20.0f && std::fabs(err) > 0.25f) ? 1.0f : 0.0f;
    }

    if (g_finished && !g_stageClosed) {
        rawThr = 0.0f;
        rawBrk = 0.65f;
        rawStr *= 0.3f;
        rawHb  = 0.0f;
    }

    if (g_countdown > 0.0f) {
        c.throttle = rawThr;
        c.brake = 1.0f;
        c.rpm = lerpf(c.rpm, 1400.0f + rawThr * 5800.0f, std::min(1.0f, dt * 12.0f));
        return;
    }

    c.throttle   = lerpf(c.throttle,   rawThr, std::min(1.0f, dt * 16.0f));
    c.brake      = lerpf(c.brake,      rawBrk, std::min(1.0f, dt * 18.0f));
    c.steerInput = lerpf(c.steerInput, rawStr, std::min(1.0f, dt * 12.0f));
    c.handbrake  = lerpf(c.handbrake,  rawHb,  std::min(1.0f, dt * 22.0f));

    // Speed-sensitive steering lock (up to 32 deg at low speed, gentler at high speed)
    float maxSteer = 0.56f / (1.0f + std::fabs(c.vx) * 0.022f);
    c.steerAngle = c.steerInput * maxSteer;

    // 2. Query road surface & terrain beneath car
    RoadQuery rq;
    float groundY = worldHeightAt(c.pos.x, c.pos.z, c.nearestNode, &rq);
    c.nearestNode = rq.idx;
    float offRoad = clampf((std::fabs(rq.lateral) - (rq.halfW + CURB_W)) / 8.0f, 0.0f, 1.0f);
    float mu = g_surfaceGrip * (1.0f - 0.28f * offRoad);

    // 3. WRC Rally1 Hybrid System (Regen under braking/handbrake/coasting, Boost on high throttle)
    bool regenCond = (c.brake > 0.12f || c.handbrake > 0.2f || (c.throttle < 0.08f && std::fabs(c.vx) > 10.0f));
    if (regenCond && std::fabs(c.vx) > 3.5f) {
        float rate = (c.handbrake > 0.2f) ? 0.32f : (c.brake > 0.12f ? 0.26f : 0.10f);
        c.hybridEnergy = clampf(c.hybridEnergy + rate * dt, 0.0f, 1.0f);
        if (c.hybridEnergy >= 0.42f) c.hybridReady = true;
        c.hybridActive = false;
    } else if (c.throttle > 0.72f && c.hybridReady && c.hybridEnergy > 0.02f && c.gear >= 1) {
        c.hybridActive = true;
        c.hybridEnergy = clampf(c.hybridEnergy - 0.24f * dt, 0.0f, 1.0f);
        if (c.hybridEnergy <= 0.025f) {
            c.hybridReady = false;
            c.hybridActive = false;
        }
    } else {
        c.hybridActive = false;
    }

    // 4. Automatic 6-Speed Sequential Gearbox + Reverse (R, 1..6)
    static const float gearTop[7] = { 18.0f, 15.5f, 24.5f, 34.0f, 44.0f, 55.0f, 68.0f };
    float absVx = std::fabs(c.vx);
    if (c.vx < -0.6f || (c.vx < 1.2f && rawBrk > 0.2f && rawThr < 0.05f)) {
        c.gear = 0; // Reverse
    } else {
        if (c.gear == 0 && (rawThr > 0.1f || c.vx > 0.3f)) c.gear = 1;
        int wantGear = 6;
        for (int g = 1; g <= 6; ++g) {
            if (absVx < gearTop[g] * 0.94f) { wantGear = g; break; }
        }
        if (wantGear != c.gear && c.shiftTimer <= 0.0f) {
            c.gear = wantGear;
            c.shiftTimer = 0.12f;
        }
    }
    if (c.shiftTimer > 0.0f) c.shiftTimer = std::max(0.0f, c.shiftTimer - dt);

    float gLow = (c.gear <= 1) ? 0.0f : gearTop[c.gear - 1] * 0.68f;
    float gHigh = gearTop[c.gear];
    float gNorm = clampf((absVx - gLow) / std::max(4.0f, gHigh - gLow), 0.0f, 1.0f);
    float targetRpm = 1350.0f + gNorm * 6600.0f + c.throttle * (c.airborne ? 2400.0f : 750.0f) + c.wheelSpin * 900.0f;
    c.rpm = lerpf(c.rpm, clampf(targetRpm, 1100.0f, 8800.0f), std::min(1.0f, dt * 16.0f));

    // 5. Four-wheel / Bicycle Dynamics in Car Frame (+X = forward, +Y = left)
    Vec3 fwd(std::sin(c.heading), 0, std::cos(c.heading));
    Vec3 left(std::cos(c.heading), 0, -std::sin(c.heading));

    if (!c.airborne) {
        const float lf = 1.28f, lr = 1.27f, wheelbase = lf + lr;
        float vxSafe = std::max(4.5f, std::fabs(c.vx)) * (c.vx >= 0.0f ? 1.0f : -1.0f);

        float alphaF = std::atan2(c.vy + lf * c.yawRate, vxSafe) - c.steerAngle * (c.vx >= 0.0f ? 1.0f : -1.0f);
        float alphaR = std::atan2(c.vy - lr * c.yawRate, vxSafe);

        // Smooth nonlinear tire lateral force (saturates smoothly for controllable rally drifts)
        float cf = 5.2f, cr = 5.6f * (1.0f - 0.58f * c.handbrake);
        float fyF = -mu * 11.2f * std::tanh(cf * alphaF);
        float fyR = -mu * 11.8f * (1.0f - 0.48f * c.handbrake) * std::tanh(cr * alphaR);

        // Longitudinal traction & braking
        float boost = c.hybridActive ? 1.34f : 1.0f;
        float driveAcc = 0.0f, brakeAcc = 0.0f;
        if (c.gear == 0) {
            // Reverse mode: S (rawBrk) drives backwards, W (rawThr) brakes
            driveAcc = -c.brake * 12.5f * (0.6f + 0.4f * mu);
            brakeAcc =  c.throttle * 24.0f * mu;
        } else {
            float gearFactor = 1.18f - 0.08f * (c.gear - 1);
            driveAcc = c.throttle * 21.5f * gearFactor * boost * (0.58f + 0.42f * mu);
            brakeAcc = -c.brake * 27.5f * mu * sgnf(c.vx);
            if (std::fabs(c.vx) < 0.8f && c.throttle < 0.05f) brakeAcc = -c.vx * 10.0f;
        }

        float hbDrag = -c.handbrake * 10.5f * mu * std::tanh(c.vx * 0.5f);
        float aeroDrag = -c.vx * std::fabs(c.vx) * 0.0040f - c.vx * (0.16f + 0.85f * offRoad);
        float slopeAcc = -GRAV * dot(fwd, rq.fwd) * g_nodes[rq.idx].slope * 0.85f;

        float dynBlend = smooth01((std::fabs(c.vx) - 2.0f) / 5.5f);
        float ax = driveAcc + brakeAcc + hbDrag + aeroDrag + slopeAcc - fyF * std::sin(c.steerAngle) * dynBlend;
        float ay = (fyF * std::cos(c.steerAngle) + fyR) * dynBlend - c.vy * (0.55f + 1.2f * offRoad);

        // Integrate velocities in body frame
        c.vx += (ax + c.vy * c.yawRate * dynBlend) * dt;
        c.vy = lerpf(c.vy * 0.82f, c.vy + (ay - c.vx * c.yawRate) * dt, dynBlend);
        c.vx = clampf(c.vx, -18.0f, 67.0f);

        float ackermannYaw = (c.vx / wheelbase) * std::tan(c.steerAngle);
        float yawAcc = (lf * fyF * std::cos(c.steerAngle) - lr * fyR) * 0.58f - c.yawRate * 2.1f;
        float dynYaw = c.yawRate + yawAcc * dt;
        if (std::fabs(c.vx) < 0.6f && c.throttle < 0.05f && c.brake < 0.05f) {
            c.vx *= 0.90f; c.vy *= 0.80f; c.yawRate *= 0.80f;
        } else {
            c.yawRate = lerpf(ackermannYaw, dynYaw, dynBlend);
        }
        c.yawRate = clampf(c.yawRate, -3.6f, 3.6f);
        c.heading = wrapPi(c.heading + c.yawRate * dt);

        c.ax = lerpf(c.ax, ax, std::min(1.0f, dt * 14.0f));
        c.ay = lerpf(c.ay, ay, std::min(1.0f, dt * 14.0f));
        c.wheelSpin = clampf((c.throttle * 1.3f - mu) * 0.8f + std::fabs(alphaR) * 0.6f * dynBlend + c.handbrake * 0.7f, 0.0f, 1.0f);
    } else {
        // Airborne: preserve momentum, slight air drag
        c.vx *= (1.0f - 0.06f * dt);
        c.vy *= (1.0f - 0.06f * dt);
        c.heading = wrapPi(c.heading + c.yawRate * 0.45f * dt);
        c.ax = lerpf(c.ax, 0.0f, dt * 5.0f);
        c.ay = lerpf(c.ay, 0.0f, dt * 5.0f);
    }

    // Recompute horizontal basis and move in world XZ
    fwd  = Vec3(std::sin(c.heading), 0, std::cos(c.heading));
    left = Vec3(std::cos(c.heading), 0, -std::sin(c.heading));
    c.pos.x += (fwd.x * c.vx + left.x * c.vy) * dt;
    c.pos.z += (fwd.z * c.vx + left.z * c.vy) * dt;
    c.speed = c.vx;
    c.lateralSpeed = c.vy;

    // Resolve collisions with solid trees, boulders, buildings, grandstands, haybales
    resolveSceneryCollisions(c, fwd, left);

    // Gently keep car inside outer boundary corridor (+/- 38m from road)
    groundY = worldHeightAt(c.pos.x, c.pos.z, c.nearestNode, &rq);
    c.nearestNode = rq.idx;
    if (std::fabs(rq.lateral) > 38.0f) {
        Vec3 pull = (rq.closest - c.pos).normalized();
        c.pos.x += pull.x * 12.0f * dt;
        c.pos.z += pull.z * 12.0f * dt;
        c.vx *= 0.96f;
    }

    // 6. Analytical Crest Jump (airborne) & 4-Corner Road Attitude
    float newGroundVy = clampf((groundY - c.prevGroundY) / dt, -25.0f, 25.0f);
    c.groundVy = lerpf(c.groundVy, newGroundVy, std::min(1.0f, dt * 18.0f));
    c.prevGroundY = groundY;

    int prevNode = (rq.idx - 2 + N) % N;
    int nextNode = (rq.idx + 3) % N;
    if (!g_stageClosed) {
        prevNode = std::max(0, rq.idx - 2);
        nextNode = std::min(N - 1, rq.idx + 3);
    }
    float crestDrop = g_nodes[prevNode].slope - g_nodes[nextNode].slope;
    c.groundAccel = -crestDrop * c.vx * c.vx / 17.5f;

    if (!c.airborne) {
        // Launch airborne ONLY on real high-speed jump crests where road slope drops sharply
        if (crestDrop > 0.085f && c.vx > 22.0f && std::fabs(rq.lateral) < rq.halfW + 2.5f && dot(fwd, rq.fwd) > 0.6f) {
            c.airborne = true;
            c.vertVel = std::max(2.2f, c.vx * g_nodes[prevNode].slope * 0.92f + 1.6f);
            c.airTime = 0.0f;
        } else {
            c.pos.y = groundY;
            c.vertVel = c.groundVy;
        }
    } else {
        c.airTime += dt;
        c.vertVel -= GRAV * 1.45f * dt;
        c.pos.y += c.vertVel * dt;
        if (c.pos.y <= groundY) {
            c.pos.y = groundY;
            c.airborne = false;
            float impact = std::min(3.5f, std::fabs(c.vertVel - c.groundVy) * 0.25f);
            c.suspVel -= impact;
            g_shake = std::min(1.0f, g_shake + impact * 0.22f);
            c.vertVel = c.groundVy;
        }
    }

    // Sample 4 wheel ground heights for geometric pitch & roll attitude
    float hF = worldHeightAt(c.pos.x + fwd.x * AXLE_F, c.pos.z + fwd.z * AXLE_F, c.nearestNode);
    float hB = worldHeightAt(c.pos.x + fwd.x * AXLE_R, c.pos.z + fwd.z * AXLE_R, c.nearestNode);
    float hL = worldHeightAt(c.pos.x + left.x * WHEEL_X, c.pos.z + left.z * WHEEL_X, c.nearestNode);
    float hR = worldHeightAt(c.pos.x - left.x * WHEEL_X, c.pos.z - left.z * WHEEL_X, c.nearestNode);
    float targetAttPitch = std::atan2(hF - hB, AXLE_F - AXLE_R);
    float targetAttRoll  = std::atan2(hL - hR, 2.0f * WHEEL_X);
    if (c.airborne) {
        targetAttPitch = clampf(c.vertVel * 0.018f, -0.22f, 0.22f);
        targetAttRoll *= 0.3f;
    }
    c.attPitch = lerpf(c.attPitch, targetAttPitch, std::min(1.0f, dt * 14.0f));
    c.attRoll  = lerpf(c.attRoll,  targetAttRoll,  std::min(1.0f, dt * 14.0f));

    // Spring-damper suspension bounce, body roll & body pitch from weight transfer
    float bump = (!c.airborne) ? (std::sin(c.pos.x * 2.1f + c.pos.z * 2.3f) * 0.018f * clampf(absVx / 18.0f, 0.0f, 1.0f)) : 0.0f;
    c.suspVel += (-130.0f * (c.suspBounce - bump) - 16.0f * c.suspVel) * dt;
    c.suspBounce = clampf(c.suspBounce + c.suspVel * dt, -0.16f, 0.16f);

    float wantRoll  = clampf( c.ay * 0.0055f, -0.14f, 0.14f);
    float wantPitch = clampf(-c.ax * 0.0042f, -0.12f, 0.12f);
    c.bodyRoll  = lerpf(c.bodyRoll,  wantRoll,  std::min(1.0f, dt * 10.0f));
    c.bodyPitch = lerpf(c.bodyPitch, wantPitch, std::min(1.0f, dt * 10.0f));

    c.wheelRot = std::fmod(c.wheelRot + (c.vx / WHEEL_R) * dt, 2.0f * PI);
    c.slipAngle = (absVx > 2.5f) ? std::atan2(c.vy, absVx) : 0.0f;

    // 7. Drift score & Stage finish / Lap tracking
    if (!g_finished) c.stageTime += dt;
    bool drifting = (!c.airborne && absVx > 7.5f && (std::fabs(c.slipAngle) > 0.16f || c.handbrake > 0.3f));
    if (drifting) {
        float pts = absVx * (std::fabs(c.slipAngle) + 0.25f * c.handbrake) * 14.0f * dt;
        c.driftCombo += pts;
        c.driftScore += pts;
    } else {
        c.driftCombo = std::max(0.0f, c.driftCombo - 45.0f * dt);
    }

    if (g_stageClosed) {
        int startNode = std::min(6, N - 1);
        if (g_prevNodeIdx > N - 20 && c.nearestNode < 20 && c.stageTime - g_lapStart > 12.0f) {
            float lapT = c.stageTime - g_lapStart;
            if (g_bestLap <= 0.0f || lapT < g_bestLap) g_bestLap = lapT;
            g_lapStart = c.stageTime;
            ++g_lap;
        }
        (void)startNode;
    } else if (!g_finished && c.nearestNode >= N - 8) {
        g_finished = true;
        g_finishTime = c.stageTime;
    }
    g_prevNodeIdx = c.nearestNode;
}

static void updateSimulation(float frameDt) {
    frameDt = clampf(frameDt, 0.001f, 0.05f);
    g_timeAccum += frameDt;
    pollJoystick();

    if (g_countdown > 0.0f) {
        g_countdown -= frameDt;
        if (g_countdown <= 0.0f) { g_countdown = 0.0f; g_goFlash = 1.2f; }
    }
    if (g_goFlash > 0.0f) g_goFlash = std::max(0.0f, g_goFlash - frameDt);
    if (g_shake > 0.0f)   g_shake   = std::max(0.0f, g_shake - frameDt * 2.4f);

    // Fixed 240 Hz physics sub-stepping
    int steps = std::max(1, std::min(16, (int)std::ceil(frameDt / PHYS_DT)));
    float subDt = frameDt / steps;
    for (int i = 0; i < steps; ++i) stepPhysics(subDt);

    RallyCar& c = g_car;

    // Dynamic brake rotor heat simulation: increases on hard braking, cools with air speed
    if (c.brake > 0.12f && std::fabs(c.vx) > 3.0f) {
        c.brakeHeat = std::min(1.0f, c.brakeHeat + frameDt * (0.85f * c.brake + 0.35f * c.handbrake));
    } else {
        c.brakeHeat = std::max(0.0f, c.brakeHeat - frameDt * (0.18f + 0.012f * std::fabs(c.vx)));
    }

    // Dynamic anti-lag (ALS) exhaust backfire: trigger on gear shift or abrupt throttle lift-off at high RPM
    if (c.shiftTimer > 0.06f || (c.prevThrottle > 0.55f && c.throttle < 0.15f && c.rpm > 5200.0f)) {
        if (c.backfireTimer <= 0.0f) {
            c.backfireTimer = randRange(0.08f, 0.18f);
            c.backfireIntensity = randRange(0.75f, 1.0f);
            Vec3 fwdSp(std::sin(c.heading), 0, std::cos(c.heading));
            Vec3 leftSp(std::cos(c.heading), 0, -std::sin(c.heading));
            Vec3 exhPos = c.pos + leftSp * 0.42f - fwdSp * 2.12f + Vec3(0, 0.24f, 0);
            spawnSparks(exhPos, -fwdSp);
        }
    }
    c.prevThrottle = c.throttle;
    if (c.backfireTimer > 0.0f) c.backfireTimer = std::max(0.0f, c.backfireTimer - frameDt);

    // Windshield wiper sweep in rain
    if (CLIMATES[g_climate].rain) {
        c.wiperAngle = std::sin(g_timeAccum * 7.5f);
    } else {
        c.wiperAngle = -1.0f; // parked
    }

    // Emit 4-wheel skidmarks & tire dust/mud/snow spray once per frame
    Vec3 fwd(std::sin(c.heading), 0, std::cos(c.heading));
    Vec3 left(std::cos(c.heading), 0, -std::sin(c.heading));
    const float wx[4] = { WHEEL_X, -WHEEL_X,  WHEEL_X, -WHEEL_X};
    const float wz[4] = { AXLE_F,   AXLE_F,   AXLE_R,   AXLE_R};
    bool leaveSkid = (!c.airborne && (std::fabs(c.slipAngle) > 0.15f || c.handbrake > 0.25f || c.brake > 0.65f || c.wheelSpin > 0.45f));

    for (int w = 0; w < 4; ++w) {
        Vec3 wp = c.pos + left * wx[w] + fwd * wz[w];
        wp.y = worldHeightAt(wp.x, wp.z, c.nearestNode) + 0.04f;
        bool wheelActive = leaveSkid && (w >= 2 || std::fabs(c.slipAngle) > 0.22f || c.brake > 0.7f);
        if (wheelActive) {
            if (c.hasSkid[w]) {
                float d = (wp - c.lastSkid[w]).length();
                if (d > 0.35f && d < 6.0f) {
                    float a = clampf(0.25f + std::fabs(c.slipAngle) * 0.65f + c.handbrake * 0.25f, 0.2f, 0.72f);
                    pushSkid(c.lastSkid[w], wp, a);
                    c.lastSkid[w] = wp;
                }
            } else {
                c.lastSkid[w] = wp;
                c.hasSkid[w] = true;
            }
            if (w >= 2 && std::fabs(c.vx) > 4.0f) {
                spawnDust(wp, fwd * c.vx + left * c.vy, std::fabs(c.slipAngle) + 0.4f);
            }
        } else {
            c.hasSkid[w] = false;
        }
    }

    // Update FX particles
    for (size_t i = 0; i < g_fx.size(); ) {
        Particle3D& p = g_fx[i];
        p.life -= frameDt;
        if (p.life <= 0.0f) { g_fx[i] = g_fx.back(); g_fx.pop_back(); continue; }
        p.pos += p.vel * frameDt;
        p.vel.y -= (p.kind == 2 ? 12.0f : 3.5f) * frameDt;
        p.size += p.growth * frameDt;
        ++i;
    }

    // Update weather particles & rainstorm lightning
    const ClimateProfile& cp = CLIMATES[g_climate];
    if (cp.rain) {
        g_lightningTimer -= frameDt;
        if (g_lightningTimer <= 0.0f) {
            g_lightning = 1.0f;
            g_lightningTimer = randRange(6.0f, 15.0f);
        }
        g_lightning = std::max(0.0f, g_lightning - frameDt * 4.5f);
    } else {
        g_lightning = 0.0f;
    }

    for (Particle3D& wp : g_weather) {
        if (cp.rain) {
            wp.pos.y -= 30.0f * frameDt;
            wp.pos.x -= 4.0f * frameDt;
        } else if (cp.snow) {
            wp.pos.y -= 7.2f * frameDt;
            wp.pos.x += std::sin(g_timeAccum * 1.8f + wp.pos.z * 0.15f) * 3.0f * frameDt;
            wp.pos.z += std::cos(g_timeAccum * 1.5f + wp.pos.x * 0.15f) * 3.0f * frameDt;
        } else {
            wp.pos.y -= 1.2f * frameDt;
        }
        if (wp.pos.y < c.pos.y - 2.0f) {
            wp.pos.y = c.pos.y + 30.0f;
            wp.pos.x = c.pos.x + randRange(-50.0f, 50.0f);
            wp.pos.z = c.pos.z + randRange(-50.0f, 50.0f);
        }
        if (std::fabs(wp.pos.x - c.pos.x) > 55.0f) wp.pos.x = c.pos.x + randRange(-50.0f, 50.0f);
        if (std::fabs(wp.pos.z - c.pos.z) > 55.0f) wp.pos.z = c.pos.z + randRange(-50.0f, 50.0f);
    }

    // Smooth camera tracking
    float velYaw = (std::fabs(c.vx) > 2.5f) ? std::atan2(fwd.x * c.vx + left.x * c.vy, fwd.z * c.vx + left.z * c.vy) : c.heading;
    float yawDiff = wrapPi(velYaw - g_camYaw);
    g_camYaw = wrapPi(g_camYaw + yawDiff * std::min(1.0f, frameDt * 6.5f));
    Vec3 camDir(std::sin(g_camYaw), 0, std::cos(g_camYaw));

    Vec3 shakeVec(randRange(-1.0f, 1.0f) * g_shake * 0.18f,
                  randRange(-1.0f, 1.0f) * g_shake * 0.18f,
                  randRange(-1.0f, 1.0f) * g_shake * 0.18f);

    if (g_cameraMode == 0) {
        Vec3 wantPos = c.pos - camDir * 7.2f + Vec3(0, 2.75f, 0);
        wantPos.y = std::max(wantPos.y, worldHeightAt(wantPos.x, wantPos.z, c.nearestNode) + 1.2f);
        Vec3 wantTgt = c.pos + fwd * 4.5f + Vec3(0, 1.05f, 0);
        g_camPos    = g_camPos    + (wantPos - g_camPos)    * std::min(1.0f, frameDt * 11.0f) + shakeVec;
        g_camTarget = g_camTarget + (wantTgt - g_camTarget) * std::min(1.0f, frameDt * 15.0f);
    } else if (g_cameraMode == 1) {
        Vec3 wantPos = c.pos - camDir * 5.0f + Vec3(0, 1.75f, 0);
        wantPos.y = std::max(wantPos.y, worldHeightAt(wantPos.x, wantPos.z, c.nearestNode) + 0.9f);
        Vec3 wantTgt = c.pos + fwd * 7.0f + Vec3(0, 0.85f, 0);
        g_camPos    = g_camPos    + (wantPos - g_camPos)    * std::min(1.0f, frameDt * 14.0f) + shakeVec;
        g_camTarget = g_camTarget + (wantTgt - g_camTarget) * std::min(1.0f, frameDt * 18.0f);
    } else if (g_cameraMode == 2) {
        g_camPos    = c.pos + fwd * 0.35f + Vec3(0, 1.38f + c.suspBounce, 0) + shakeVec * 0.4f;
        g_camTarget = g_camPos + fwd * 20.0f + Vec3(0, -0.6f - c.attPitch * 12.0f, 0);
    } else {
        Vec3 wantPos = c.pos - camDir * 17.0f + Vec3(5.0f, 13.5f, 0);
        Vec3 wantTgt = c.pos + fwd * 4.0f;
        g_camPos    = g_camPos    + (wantPos - g_camPos)    * std::min(1.0f, frameDt * 6.0f);
        g_camTarget = g_camTarget + (wantTgt - g_camTarget) * std::min(1.0f, frameDt * 10.0f);
    }
}

// ============================================================================
//  STATE_DRAW_MAP: 2D WHITE ENGINEERING GRID CANVAS
// ============================================================================
static void updateCanvasPreview() {
    if (!g_previewDirty) return;
    g_previewDirty = false;
    g_previewLine.clear();
    g_previewLen = 0.0f;
    g_previewMinR = 999.0f;
    if (g_rawPts.size() < 2) return;

    Vec2 c; float mpp;
    canvasTransform(g_rawPts, c, mpp);
    g_previewScale = mpp;
    std::vector<Vec2> world;
    for (const Vec2& p : g_rawPts) world.push_back(Vec2((p.x - c.x) * mpp, (p.y - c.y) * mpp));
    std::vector<Vec2> smoothW = buildCenterline(world, g_closed);
    int n = (int)smoothW.size();
    if (n < 2) return;

    for (int i = 0; i < n; ++i) {
        g_previewLine.push_back(Vec2(c.x + smoothW[i].x / mpp, c.y + smoothW[i].y / mpp));
        if (i > 0) g_previewLen += len2(smoothW[i] - smoothW[i - 1]);
        if (i > 0 && i < n - 1) {
            Vec2 a = smoothW[i] - smoothW[i - 1], b = smoothW[i + 1] - smoothW[i];
            float ang = std::fabs(std::atan2(a.x * b.y - a.y * b.x, a.x * b.x + a.y * b.y));
            if (ang > 1e-4f) g_previewMinR = std::min(g_previewMinR, STAGE_STEP / ang);
        }
    }
    if (g_closed) g_previewLen += len2(smoothW.front() - smoothW.back());
}

static void renderStateDrawMap() {
    updateCanvasPreview();
    glViewport(0, 0, g_winW, g_winH);
    glClearColor(0.97f, 0.98f, 0.99f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    beginOrtho();

    // 1. Engineering millimeter / meter blueprint grid
    glBegin(GL_LINES);
    for (int x = 0; x < g_winW; x += 25) {
        if (x % 100 == 0) glColor4f(0.70f, 0.77f, 0.86f, 0.90f);
        else              glColor4f(0.86f, 0.90f, 0.94f, 0.80f);
        glVertex2f((float)x, 0.0f); glVertex2f((float)x, (float)g_winH);
    }
    for (int y = 0; y < g_winH; y += 25) {
        if (y % 100 == 0) glColor4f(0.70f, 0.77f, 0.86f, 0.90f);
        else              glColor4f(0.86f, 0.90f, 0.94f, 0.80f);
        glVertex2f(0.0f, (float)y); glVertex2f((float)g_winW, (float)y);
    }
    glEnd();

    float cx = g_winW * 0.5f, cy = g_winH * 0.5f;
    glLineWidth(1.6f);
    glColor4f(0.52f, 0.62f, 0.74f, 0.9f);
    glBegin(GL_LINES);
    glVertex2f(cx, 0); glVertex2f(cx, (float)g_winH);
    glVertex2f(0, cy); glVertex2f((float)g_winW, cy);
    glEnd();
    glLineWidth(1.0f);
    drawText(cx + 8, cy + 6, "MERKEZ (0M, 0M)  |  DUNYA OLCEGI: [-900M .. +900M]", 1.6f, 0.32f, 0.42f, 0.56f, 0.9f, false);

    // 2. Smooth Catmull-Rom Road Ribbon Preview & Raw Control Points
    if (g_previewLine.size() >= 2) {
        int n = (int)g_previewLine.size();
        float halfRoadPx = std::max(4.0f, 5.5f / std::max(0.2f, g_previewScale));
        // Road shoulder & asphalt ribbon
        glColor4f(0.76f, 0.80f, 0.86f, 0.85f);
        glBegin(GL_QUAD_STRIP);
        int lim = g_closed ? n : n - 1;
        for (int i = 0; i <= lim; ++i) {
            Vec2 p = g_previewLine[i % n];
            Vec2 prev = g_previewLine[(i - 1 + n) % n], next = g_previewLine[(i + 1) % n];
            Vec2 t = next - prev;
            float l = std::max(1e-4f, len2(t));
            Vec2 nrm(-t.y / l, t.x / l);
            glVertex2f(p.x - nrm.x * halfRoadPx * 1.6f, p.y - nrm.y * halfRoadPx * 1.6f);
            glVertex2f(p.x + nrm.x * halfRoadPx * 1.6f, p.y + nrm.y * halfRoadPx * 1.6f);
        }
        glEnd();

        glLineWidth(3.5f);
        glColor4f(0.86f, 0.14f, 0.14f, 0.96f);
        glBegin(g_closed ? GL_LINE_LOOP : GL_LINE_STRIP);
        for (const Vec2& p : g_previewLine) glVertex2f(p.x, p.y);
        glEnd();
        glLineWidth(1.0f);

        // Start & Finish markers
        Vec2 st = g_previewLine.front(), en = g_previewLine.back();
        drawRect(st.x - 7, st.y - 7, st.x + 7, st.y + 7, 0.10f, 0.72f, 0.24f, 1.0f);
        drawText(st.x + 11, st.y - 5, "START", 1.8f, 0.08f, 0.50f, 0.18f, 1.0f, false);
        if (!g_closed) {
            drawRect(en.x - 7, en.y - 7, en.x + 7, en.y + 7, 0.86f, 0.14f, 0.14f, 1.0f);
            drawText(en.x + 11, en.y - 5, "FINISH", 1.8f, 0.72f, 0.10f, 0.10f, 1.0f, false);
        }
    }

    if (!g_rawPts.empty()) {
        glPointSize(4.5f);
        glColor4f(0.14f, 0.30f, 0.62f, 0.85f);
        glBegin(GL_POINTS);
        for (const Vec2& p : g_rawPts) glVertex2f(p.x, p.y);
        glEnd();
        glPointSize(1.0f);
    }

    // 3. Header Banner
    drawRect(18, 14, g_winW - 18.0f, 108, 0.09f, 0.12f, 0.18f, 0.94f);
    drawRectLine(18, 14, g_winW - 18.0f, 108, 2.0f, 0.88f, 0.18f, 0.14f, 1.0f);
    drawText(34, 26, "WRC RALLY1 HYBRID SIMULATOR  -  2D MUHENDISLIK ETAP CIZIM EKRANI (STATE_DRAW_MAP)", 2.1f, 0.98f, 0.98f, 1.0f);
    drawText(34, 54, "[SOL TIK + SURUKLE]: Serbest El Rota Ciz   |   [L]: Kapali Devre Pist (Circuit)   |   [C / SAG TIK]: Tuvali Temizle", 1.7f, 0.78f, 0.86f, 0.96f);
    drawText(34, 80, "[ENTER]: Cizimi (-900M..+900M) 3D Dunyaya Olcekle, Catmull-Rom Spline & Rastgele Iklim ile Surusu Baslat", 1.7f, 0.30f, 0.95f, 0.55f);

    // 4. Footer Telemetry Bar
    drawRect(18, g_winH - 52.0f, g_winW - 18.0f, g_winH - 14.0f, 0.10f, 0.14f, 0.20f, 0.92f);
    char info[256];
    std::snprintf(info, sizeof(info),
                  "NOKTA: %zu  |  ETAP UZUNLUGU: %.0f M  |  MIN VIRAJ YARICAPI: %.0f M  |  MOD: %s  |  (Bos iken ENTER = Ornek Pist)",
                  g_rawPts.size(), g_previewLen, g_previewMinR > 500 ? 0.0f : g_previewMinR,
                  g_closed ? "KAPALI DEVRE [L]" : "ACIK ETAP");
    drawText(32, g_winH - 38.0f, info, 1.7f, 0.92f, 0.95f, 0.98f);

    endOrtho();
}

// ============================================================================
//  3D PARTICLES, SKIDS, HUD & PACENOTE BADGES
// ============================================================================
static void renderSkidsAndParticles() {
    // 1. Tire Skid Marks on road
    if (!g_skids.empty()) {
        glDisable(GL_LIGHTING);
        texOff();
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        const ClimateProfile& cp = CLIMATES[g_climate];
        float sr = cp.snow ? 0.55f : (cp.rain ? 0.16f : 0.20f);
        float sg = cp.snow ? 0.62f : (cp.rain ? 0.12f : 0.16f);
        float sb = cp.snow ? 0.72f : (cp.rain ? 0.09f : 0.12f);

        glBegin(GL_QUADS);
        for (const SkidMark3D& s : g_skids) {
            Vec3 d = s.b - s.a;
            Vec3 r = cross(d, Vec3(0, 1, 0)).normalized() * 0.14f;
            glColor4f(sr, sg, sb, s.alpha);
            glV(s.a - r); glV(s.a + r); glV(s.b + r); glV(s.b - r);
        }
        glEnd();
        glDisable(GL_BLEND);
        glEnable(GL_LIGHTING);
    }

    // 2. Billboarded Dust/Mud/Snow Spray & Collision Sparks
    float mv[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, mv);
    Vec3 camR(mv[0], mv[4], mv[8]), camU(mv[1], mv[5], mv[9]);

    glDisable(GL_LIGHTING);
    glDepthMask(GL_FALSE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    texOn(g_texSoft);
    glBegin(GL_QUADS);
    for (const Particle3D& p : g_fx) {
        float t = clampf(p.life / p.maxLife, 0.0f, 1.0f);
        glColor4f(p.color[0], p.color[1], p.color[2], p.color[3] * t);
        Vec3 r = camR * p.size, u = camU * p.size;
        glTexCoord2f(0, 0); glV(p.pos - r - u);
        glTexCoord2f(1, 0); glV(p.pos + r - u);
        glTexCoord2f(1, 1); glV(p.pos + r + u);
        glTexCoord2f(0, 1); glV(p.pos - r + u);
    }
    glEnd();
    texOff();

    // 3. Climate Precipitation (Rain Streaks / Blizzard Snowflakes)
    const ClimateProfile& cp = CLIMATES[g_climate];
    if (cp.rain) {
        glLineWidth(1.5f);
        glColor4f(0.74f, 0.82f, 0.94f, 0.55f);
        glBegin(GL_LINES);
        for (const Particle3D& wp : g_weather) {
            glV(wp.pos);
            glVertex3f(wp.pos.x - 0.22f, wp.pos.y - 1.65f, wp.pos.z);
        }
        glEnd();
        glLineWidth(1.0f);
    } else if (cp.snow) {
        texOn(g_texSoft);
        glColor4f(0.95f, 0.97f, 1.0f, 0.85f);
        glBegin(GL_QUADS);
        Vec3 r = camR * 0.16f, u = camU * 0.16f;
        for (const Particle3D& wp : g_weather) {
            glTexCoord2f(0, 0); glV(wp.pos - r - u);
            glTexCoord2f(1, 0); glV(wp.pos + r - u);
            glTexCoord2f(1, 1); glV(wp.pos + r + u);
            glTexCoord2f(0, 1); glV(wp.pos - r + u);
        }
        glEnd();
        texOff();
    }

    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glEnable(GL_LIGHTING);
}

static void renderHUD(bool haveSunFlare = false, float sunScreenX = 0, float sunScreenY = 0, float sunDot = 0) {
    beginOrtho();
    const ClimateProfile& cp = CLIMATES[g_climate];

    // 0A. Cinematic Filmic Corner Vignette (smooth corner darkening)
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    float vw = g_winW * 0.28f, vh = g_winH * 0.28f;
    glBegin(GL_QUADS);
    // Top-left corner
    glColor4f(0.0f, 0.0f, 0.02f, 0.40f); glVertex2f(0.0f, 0.0f);
    glColor4f(0.0f, 0.0f, 0.02f, 0.00f); glVertex2f(vw,   0.0f);
    glColor4f(0.0f, 0.0f, 0.02f, 0.00f); glVertex2f(vw,   vh);
    glColor4f(0.0f, 0.0f, 0.02f, 0.00f); glVertex2f(0.0f, vh);
    // Top-right corner
    glColor4f(0.0f, 0.0f, 0.02f, 0.00f); glVertex2f(g_winW - vw, 0.0f);
    glColor4f(0.0f, 0.0f, 0.02f, 0.40f); glVertex2f(g_winW,      0.0f);
    glColor4f(0.0f, 0.0f, 0.02f, 0.00f); glVertex2f(g_winW,      vh);
    glColor4f(0.0f, 0.0f, 0.02f, 0.00f); glVertex2f(g_winW - vw, vh);
    // Bottom-left corner
    glColor4f(0.0f, 0.0f, 0.02f, 0.00f); glVertex2f(0.0f, g_winH - vh);
    glColor4f(0.0f, 0.0f, 0.02f, 0.00f); glVertex2f(vw,   g_winH - vh);
    glColor4f(0.0f, 0.0f, 0.02f, 0.00f); glVertex2f(vw,   g_winH);
    glColor4f(0.0f, 0.0f, 0.02f, 0.40f); glVertex2f(0.0f, g_winH);
    // Bottom-right corner
    glColor4f(0.0f, 0.0f, 0.02f, 0.00f); glVertex2f(g_winW - vw, g_winH - vh);
    glColor4f(0.0f, 0.0f, 0.02f, 0.00f); glVertex2f(g_winW,      g_winH - vh);
    glColor4f(0.0f, 0.0f, 0.02f, 0.40f); glVertex2f(g_winW,      g_winH);
    glColor4f(0.0f, 0.0f, 0.02f, 0.00f); glVertex2f(g_winW - vw, g_winH);
    glEnd();

    // 0B. Dynamic Sun Lens Flare & Chromatic Ghost Artifacts
    if (haveSunFlare && sunDot > 0.45f && cp.sunGlow > 0.0f) {
        float flareIntensity = clampf((sunDot - 0.45f) / 0.48f, 0.0f, 1.0f) * cp.sunGlow;
        float scrCX = g_winW * 0.5f, scrCY = g_winH * 0.5f;
        float fdx = scrCX - sunScreenX, fdy = scrCY - sunScreenY;

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);

        // A. Central warm sun corona disc
        texOn(g_texSoft);
        float cr = 140.0f * flareIntensity;
        glColor4f(cp.sunColor[0], cp.sunColor[1], cp.sunColor[2], 0.65f * flareIntensity);
        glBegin(GL_QUADS);
        glTexCoord2f(0, 0); glVertex2f(sunScreenX - cr, sunScreenY - cr);
        glTexCoord2f(1, 0); glVertex2f(sunScreenX + cr, sunScreenY - cr);
        glTexCoord2f(1, 1); glVertex2f(sunScreenX + cr, sunScreenY + cr);
        glTexCoord2f(0, 1); glVertex2f(sunScreenX - cr, sunScreenY + cr);
        glEnd();

        // B. Anamorphic horizontal lens flare streak
        glColor4f(0.85f, 0.92f, 1.0f, 0.42f * flareIntensity);
        glBegin(GL_QUADS);
        glTexCoord2f(0, 0); glVertex2f(sunScreenX - cr * 4.5f, sunScreenY - 10.0f);
        glTexCoord2f(1, 0); glVertex2f(sunScreenX + cr * 4.5f, sunScreenY - 10.0f);
        glTexCoord2f(1, 1); glVertex2f(sunScreenX + cr * 4.5f, sunScreenY + 10.0f);
        glTexCoord2f(0, 1); glVertex2f(sunScreenX - cr * 4.5f, sunScreenY + 10.0f);
        glEnd();

        // C. Chromatic ghost reflections along flare axis
        static const float ghostT[5] = {0.35f, 0.72f, 1.15f, 1.48f, 1.85f};
        static const float ghostR[5] = {32.0f, 48.0f, 24.0f, 65.0f, 38.0f};
        static const float ghostCol[5][3] = {
            {0.25f, 0.85f, 0.95f}, // cyan
            {0.95f, 0.80f, 0.20f}, // amber
            {0.85f, 0.30f, 0.95f}, // magenta
            {0.35f, 0.95f, 0.45f}, // green
            {0.95f, 0.92f, 0.70f}  // gold
        };
        for (int g = 0; g < 5; ++g) {
            float gx = sunScreenX + fdx * ghostT[g];
            float gy = sunScreenY + fdy * ghostT[g];
            float gr = ghostR[g] * (0.6f + 0.4f * flareIntensity);
            glColor4f(ghostCol[g][0], ghostCol[g][1], ghostCol[g][2], 0.28f * flareIntensity);
            glBegin(GL_QUADS);
            glTexCoord2f(0, 0); glVertex2f(gx - gr, gy - gr);
            glTexCoord2f(1, 0); glVertex2f(gx + gr, gy - gr);
            glTexCoord2f(1, 1); glVertex2f(gx + gr, gy + gr);
            glTexCoord2f(0, 1); glVertex2f(gx - gr, gy + gr);
            glEnd();
        }
        texOff();
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }

    // 0C. High-Speed Radial Motion Streaks (>110 KM/H or Hybrid Boost)
    float kmhVal = std::fabs(g_car.speed) * 3.6f;
    if (kmhVal > 110.0f || g_car.hybridActive) {
        float streakAlpha = clampf((kmhVal - 110.0f) / 65.0f, 0.0f, 0.48f);
        if (g_car.hybridActive) streakAlpha = std::max(streakAlpha, 0.60f);

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        glLineWidth(2.2f);
        float cx = g_winW * 0.5f, cy = g_winH * 0.5f;
        glBegin(GL_LINES);
        for (int i = 0; i < 24; ++i) {
            float ang = 2.0f * PI * i / 24.0f + std::sin(g_timeAccum * 35.0f + i) * 0.05f;
            float cosA = std::cos(ang), sinA = std::sin(ang);
            float rOuter = std::sqrt(cx * cx + cy * cy) * 1.05f;
            float rInner = rOuter - (120.0f + 90.0f * streakAlpha + std::sin(g_timeAccum * 40.0f + i * 2.0f) * 45.0f);
            if (g_car.hybridActive) glColor4f(0.25f, 0.85f, 1.0f, streakAlpha * 0.75f);
            else                   glColor4f(0.95f, 0.98f, 1.0f, streakAlpha * 0.55f);
            glVertex2f(cx + cosA * rOuter, cy + sinA * rOuter);
            glColor4f(1.0f, 1.0f, 1.0f, 0.0f);
            glVertex2f(cx + cosA * rInner, cy + sinA * rInner);
        }
        glEnd();
        glLineWidth(1.0f);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }

    // 0D. Dynamic Rain Droplets & Trickling Beads on Camera Lens
    if (cp.rain) {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        for (int d = 0; d < 38; ++d) {
            uint32_t hd = (uint32_t)(hashf(d, 42, 9999u) * 65536.0f);
            float bx = std::fmod((float)(hd % (uint32_t)g_winW), (float)g_winW);
            float by = std::fmod((float)((hd / 13) % (uint32_t)g_winH) + g_timeAccum * (18.0f + (d % 5) * 8.0f), (float)g_winH);
            float dr = 2.5f + (d % 4) * 1.5f;

            // Water bead body (semi-transparent refractive gray-blue)
            drawRect(bx - dr, by - dr, bx + dr, by + dr * 1.4f, 0.72f, 0.82f, 0.92f, 0.38f);
            // Droplet specular glint (white dot)
            drawRect(bx - dr * 0.4f, by - dr * 0.4f, bx + dr * 0.2f, by, 1.0f, 1.0f, 1.0f, 0.75f);
            // Trickle trail behind droplet
            if (d % 3 == 0) {
                float trLen = 14.0f + (d % 5) * 6.0f;
                drawRect(bx - 0.7f, by - trLen, bx + 0.7f, by, 0.70f, 0.80f, 0.92f, 0.22f);
            }
        }
    }

    static const char* carNames[3] = {
        "TOYOTA GR YARIS RALLY1 HYBRID",
        "HYUNDAI I20 N RALLY1 HYBRID",
        "M-SPORT FORD PUMA RALLY1 HYBRID"
    };

    // 1. Top-Left Stage & Climate Telemetry Panel
    drawRect(18, 16, 492, 118, 0.06f, 0.08f, 0.13f, 0.86f);
    drawRectLine(18, 16, 492, 118, 2.0f, 0.88f, 0.18f, 0.14f, 0.95f);

    int mins = (int)(g_car.stageTime / 60.0f);
    float secs = std::fmod(g_car.stageTime, 60.0f);
    char l1[160], l2[160], l3[160];
    std::snprintf(l1, sizeof(l1), "SURE: %02d:%05.2f  |  DRIFT: %.0f (+%.0f)",
                  mins, secs, g_car.driftScore, g_car.driftCombo);
    std::snprintf(l2, sizeof(l2), "IKLIM: %s (%.0f%%)", cp.name, g_surfaceGrip * 100.0f);
    std::snprintf(l3, sizeof(l3), "ARAC: %s %s", carNames[g_car.model % 3], g_autopilot ? "[OTOPILOT]" : "");
    drawText(28, 28, l1, 1.55f, 0.98f, 0.98f, 1.0f);
    drawText(28, 58, l2, 1.50f, 0.35f, 0.88f, 1.0f);
    drawText(28, 86, l3, 1.45f, 0.82f, 0.88f, 0.95f);

    // 2. Top-Center Co-Driver Pacenote Badge (HP, 2, 3, JP)
    const StageEvent* nextEv = nullptr;
    float evDist = 0.0f;
    if (!g_nodes.empty() && !g_events.empty()) {
        int N = (int)g_nodes.size();
        float curD = g_nodes[g_car.nearestNode].dist;
        float bestD = 220.0f;
        for (const StageEvent& ev : g_events) {
            float d = g_nodes[ev.startIdx].dist - curD;
            if (g_stageClosed && d < -10.0f) d += g_stageLength;
            if (d >= -8.0f && d < bestD) {
                bestD = d;
                nextEv = &ev;
                evDist = std::max(0.0f, d);
            }
        }
        (void)N;
    }

    if (nextEv) {
        float bx0 = std::max(512.0f, g_winW * 0.5f - 145.0f), bx1 = bx0 + 305.0f;
        float by0 = 16.0f, by1 = 108.0f;
        float br = 0.20f, bg = 0.68f, bb = 0.26f;
        const char* code = "3";
        const char* desc = "HIZLI VIRAJ";
        if (nextEv->sev == 1)      { br = 0.88f; bg = 0.14f; bb = 0.14f; code = "HP"; desc = "KESKIN U-VIRAJ"; }
        else if (nextEv->sev == 2) { br = 0.92f; bg = 0.48f; bb = 0.10f; code = "2";  desc = "SERT VIRAJ"; }
        else if (nextEv->sev == 3) { br = 0.86f; bg = 0.74f; bb = 0.12f; code = "3";  desc = "ORTA VIRAJ"; }
        else if (nextEv->sev == 4) { br = 0.18f; bg = 0.58f; bb = 0.95f; code = "JP"; desc = "TEPE ZIPLAMA"; }

        drawRect(bx0, by0, bx1, by1, 0.06f, 0.08f, 0.13f, 0.92f);
        drawRect(bx0 + 6, by0 + 6, bx0 + 86, by1 - 6, br, bg, bb, 0.96f);
        drawRectLine(bx0, by0, bx1, by1, 2.2f, br, bg, bb, 1.0f);

        // Vector arrow inside Pacenote badge
        float acx = bx0 + 46.0f, acy = by0 + 46.0f;
        glLineWidth(3.5f);
        glColor4f(1, 1, 1, 1);
        glBegin(GL_LINES);
        if (nextEv->kind == EVT_JUMP) {
            glVertex2f(acx, acy + 20); glVertex2f(acx, acy - 18);
            glVertex2f(acx, acy - 18); glVertex2f(acx - 11, acy - 5);
            glVertex2f(acx, acy - 18); glVertex2f(acx + 11, acy - 5);
        } else {
            // dir > 0 is left turn, dir < 0 is right turn
            float s = (nextEv->dir > 0) ? -1.0f : 1.0f;
            glVertex2f(acx - s * 10, acy + 18); glVertex2f(acx - s * 10, acy - 4);
            glVertex2f(acx - s * 10, acy - 4);  glVertex2f(acx + s * 14, acy - 14);
        }
        glEnd();
        glLineWidth(1.0f);

        char t1[64], t2[64];
        std::snprintf(t1, sizeof(t1), "[%s] %s", code,
                      nextEv->kind == EVT_JUMP ? "CREST" : (nextEv->dir > 0 ? "SOL" : "SAG"));
        std::snprintf(t2, sizeof(t2), "%s (%.0fM)", desc, evDist);
        drawText(bx0 + 96, by0 + 22, t1, 2.2f, 1.0f, 0.96f, 0.86f);
        drawText(bx0 + 96, by0 + 62, t2, 1.58f, 0.82f, 0.88f, 0.96f);
    }

    // 3. Top-Right 2D Stage Mini-Map
    if (g_nodes.size() >= 4) {
        float mx0 = g_winW - 212.0f, my0 = 16.0f, mx1 = g_winW - 18.0f, my1 = 210.0f;
        drawRect(mx0, my0, mx1, my1, 0.06f, 0.08f, 0.13f, 0.84f);
        drawRectLine(mx0, my0, mx1, my1, 1.6f, 0.45f, 0.56f, 0.72f, 0.95f);
        float mcx = 0.5f * (mx0 + mx1), mcy = 0.5f * (my0 + my1), sc = 80.0f / 900.0f;
        glLineWidth(2.2f);
        glColor4f(0.78f, 0.84f, 0.92f, 0.88f);
        glBegin(g_stageClosed ? GL_LINE_LOOP : GL_LINE_STRIP);
        for (size_t i = 0; i < g_nodes.size(); i += 2)
            glVertex2f(mcx + g_nodes[i].pos.x * sc, mcy + g_nodes[i].pos.z * sc);
        glEnd();
        glLineWidth(1.0f);
        float px = mcx + g_car.pos.x * sc, pz = mcy + g_car.pos.z * sc;
        drawRect(px - 4, pz - 4, px + 4, pz + 4, 0.98f, 0.22f, 0.16f, 1.0f);
        drawText(mx0 + 10, my1 - 20, "HARITA (-900M..+900M)", 1.4f, 0.72f, 0.80f, 0.90f);
    }

    // 4. Bottom-Right WRC Rally1 Digital + Arc Tachometer, Speed, Gear & Hybrid Boost
    float gx0 = g_winW - 385.0f, gy0 = g_winH - 170.0f, gx1 = g_winW - 18.0f, gy1 = g_winH - 18.0f;
    drawRect(gx0, gy0, gx1, gy1, 0.06f, 0.08f, 0.13f, 0.90f);
    drawRectLine(gx0, gy0, gx1, gy1, 2.2f,
                 g_car.hybridActive ? 0.15f : 0.86f,
                 g_car.hybridActive ? 0.88f : 0.20f,
                 g_car.hybridActive ? 1.00f : 0.16f, 0.96f);

    // Circular RPM arc gauge on left of cluster
    float acx = gx0 + 70.0f, acy = gy0 + 82.0f;
    float rpmFrac = clampf((g_car.rpm - 1000.0f) / 7800.0f, 0.0f, 1.0f);
    drawArc(acx, acy, 42.0f, 54.0f, PI * 0.75f, PI * 2.25f, 28, 0.18f, 0.22f, 0.28f, 0.95f);
    drawArc(acx, acy, 42.0f, 54.0f, PI * 0.75f, PI * (0.75f + 1.5f * rpmFrac), 28,
            rpmFrac > 0.82f ? 0.96f : 0.20f,
            rpmFrac > 0.82f ? 0.18f : 0.88f,
            rpmFrac > 0.82f ? 0.16f : 0.42f, 1.0f);

    const char* gearStr = (g_car.gear == 0) ? "R" :
                          (g_car.gear == 1) ? "1" :
                          (g_car.gear == 2) ? "2" :
                          (g_car.gear == 3) ? "3" :
                          (g_car.gear == 4) ? "4" :
                          (g_car.gear == 5) ? "5" : "6";
    drawText(acx - 10, acy - 16, gearStr, 4.2f, 1.0f, 0.86f, 0.22f);

    int kmh = (int)std::lround(std::fabs(g_car.speed) * 3.6f);
    char spdBuf[64], rpmBuf[64], hybBuf[80];
    std::snprintf(spdBuf, sizeof(spdBuf), "%3d KMH", kmh);
    std::snprintf(rpmBuf, sizeof(rpmBuf), "%4.0f RPM  %s", g_car.rpm, g_car.airborne ? "[AIR!]" : "");
    std::snprintf(hybBuf, sizeof(hybBuf), "HYBRID: %3.0f%% [%s]",
                  g_car.hybridEnergy * 100.0f,
                  g_car.hybridActive ? "BOOST!" : (g_car.hybridReady ? "HAZIR" : "SARJ"));

    drawText(gx0 + 140, gy0 + 22, spdBuf, 3.4f, 0.98f, 0.98f, 1.0f);
    drawText(gx0 + 140, gy0 + 68, rpmBuf, 1.9f, 0.82f, 0.88f, 0.96f);

    // Hybrid Boost Bar
    float hx0 = gx0 + 140.0f, hx1 = gx1 - 16.0f, hy0 = gy0 + 104.0f, hy1 = gy0 + 128.0f;
    drawRect(hx0, hy0, hx1, hy1, 0.14f, 0.18f, 0.24f, 1.0f);
    drawRect(hx0, hy0, hx0 + (hx1 - hx0) * g_car.hybridEnergy, hy1,
             g_car.hybridActive ? 0.15f : 0.18f,
             g_car.hybridActive ? 0.92f : 0.65f, 1.0f, 1.0f);
    drawText(hx0 + 8, hy0 + 6, hybBuf, 1.38f, 0.98f, 0.98f, 1.0f);

    // 5. Bottom-Left Controls Help Bar
    drawRect(18, g_winH - 74.0f, gx0 - 14.0f, g_winH - 18.0f, 0.06f, 0.08f, 0.13f, 0.84f);
    drawText(28, g_winH - 61.0f, "[W/A/S/D]: Surus  |  [SPACE]: El Freni  |  [C]: Kamera  |  [V]: Arac  |  [R]: Yola Sifirla  |  [P]: Otopilot", 1.32f, 0.86f, 0.90f, 0.96f);
    drawText(28, g_winH - 37.0f, "[K]: Ayni Haritada Yeni Iklim & Bina Uret  |  [M]: 2D Cizim Ekranina Don  |  Joystick (/dev/input/js0) Aktif", 1.32f, 0.42f, 0.92f, 0.64f);

    // 6. Start Countdown / Stage Finish Banner
    if (g_countdown > 0.0f) {
        char cd[32];
        std::snprintf(cd, sizeof(cd), "%d", (int)std::ceil(g_countdown));
        float w = textWidth(cd, 9.0f);
        drawText((g_winW - w) * 0.5f, g_winH * 0.36f, cd, 9.0f, 1.0f, 0.22f, 0.16f);
    } else if (g_goFlash > 0.0f) {
        float w = textWidth("GO!", 8.0f);
        drawText((g_winW - w) * 0.5f, g_winH * 0.36f, "GO!", 8.0f, 0.22f, 1.0f, 0.42f, g_goFlash);
    } else if (g_finished) {
        char fin[96];
        std::snprintf(fin, sizeof(fin), "ETAP TAMAMLANDI!  SURE: %.2f SN  ([R] TEKRAR / [M] HARITA)", g_finishTime);
        float w = textWidth(fin, 2.4f);
        drawRect((g_winW - w) * 0.5f - 20, g_winH * 0.34f - 14, (g_winW + w) * 0.5f + 20, g_winH * 0.34f + 36, 0.06f, 0.08f, 0.14f, 0.92f);
        drawText((g_winW - w) * 0.5f, g_winH * 0.34f, fin, 2.4f, 0.30f, 1.0f, 0.52f);
    }

    endOrtho();
}

// ============================================================================
//  STATE_RALLY_DRIVE: FULL 3D SCENE RENDERER
// ============================================================================
static void renderStateRallyDrive() {
    const ClimateProfile& cp = CLIMATES[g_climate];
    glViewport(0, 0, g_winW, g_winH);
    glClearColor(cp.fogColor[0], cp.fogColor[1], cp.fogColor[2], 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    double aspect = (double)g_winW / (double)std::max(1, g_winH);
    double fov = 60.0 + clampf(std::fabs(g_car.speed) * 0.18f, 0.0f, 10.0f) + (g_car.hybridActive ? 5.0 : 0.0);
    gluPerspective(fov, aspect, 0.35, 3400.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(g_camPos.x, g_camPos.y, g_camPos.z,
              g_camTarget.x, g_camTarget.y, g_camTarget.z,
              0.0, 1.0, 0.0);

    // 1. Sky Dome + Sun Glow + Animated Cloud Layer (centred on camera)
    glDisable(GL_LIGHTING);
    glDisable(GL_FOG);
    glDepthMask(GL_FALSE);
    glPushMatrix();
    glTranslatef(g_camPos.x, g_camPos.y, g_camPos.z);
    if (g_listSky) glCallList(g_listSky);

    if (cp.sunGlow > 0.0f) {
        Vec3 sdir = Vec3(cp.sunDir[0], cp.sunDir[1], cp.sunDir[2]).normalized();
        Vec3 spos = sdir * 2400.0f;
        Vec3 up(0, 1, 0);
        Vec3 r = cross(up, sdir).normalized() * (260.0f * cp.sunGlow);
        Vec3 u = cross(sdir, r).normalized() * (260.0f * cp.sunGlow);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        texOn(g_texSoft);
        glColor4f(cp.sunColor[0], cp.sunColor[1], cp.sunColor[2], 0.85f);
        glBegin(GL_QUADS);
        glTexCoord2f(0, 0); glV(spos - r - u);
        glTexCoord2f(1, 0); glV(spos + r - u);
        glTexCoord2f(1, 1); glV(spos + r + u);
        glTexCoord2f(0, 1); glV(spos - r + u);
        glEnd();
        texOff();
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDisable(GL_BLEND);
    }
    glPopMatrix();
    glDepthMask(GL_TRUE);

    // 2. Setup 3D Sun + Hemisphere Fill Lighting & Exponential Atmospheric Fog
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHT1);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glLightModeli(GL_LIGHT_MODEL_COLOR_CONTROL, GL_SEPARATE_SPECULAR_COLOR);

    float flash = g_lightning * 0.65f;
    float sunPos[4]  = {cp.sunDir[0], cp.sunDir[1], cp.sunDir[2], 0.0f};
    float fillPos[4] = {-cp.sunDir[0], 0.45f, -cp.sunDir[2], 0.0f};
    float sunDif[4]  = {cp.sunColor[0] + flash, cp.sunColor[1] + flash, cp.sunColor[2] + flash, 1.0f};
    float ambCol[4]  = {cp.ambient[0] + flash * 0.4f, cp.ambient[1] + flash * 0.4f, cp.ambient[2] + flash * 0.4f, 1.0f};
    glLightfv(GL_LIGHT0, GL_POSITION, sunPos);
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  sunDif);
    glLightfv(GL_LIGHT0, GL_SPECULAR, sunDif);
    glLightfv(GL_LIGHT0, GL_AMBIENT,  ambCol);
    glLightfv(GL_LIGHT1, GL_POSITION, fillPos);
    glLightfv(GL_LIGHT1, GL_DIFFUSE,  cp.fill);

    glEnable(GL_FOG);
    glFogi(GL_FOG_MODE, GL_EXP2);
    glFogfv(GL_FOG_COLOR, cp.fogColor);
    glFogf(GL_FOG_DENSITY, cp.fogDensity);

    // 3. Render Terrain & Extruded Road Mesh Display Lists
    if (g_listTerrain) glCallList(g_listTerrain);
    if (g_listRoad)    glCallList(g_listRoad);
    texOff();

    // 4. Render Scenery Blob Shadows & Solid 3D Scenery Chunks (with distance culling)
    Vec3 viewFwd = (g_camTarget - g_camPos).normalized();
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);
    texOn(g_texShadow);
    for (const SceneChunk& ch : g_chunks) {
        Vec3 d = ch.center - g_camPos;
        float dist2 = d.x * d.x + d.z * d.z;
        if (dist2 > 520.0f * 520.0f) continue;
        if (dist2 > ch.radius * ch.radius && dot(d, viewFwd) < -ch.radius) continue;
        glCallList(ch.decals);
    }
    texOff();
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glEnable(GL_LIGHTING);

    for (const SceneChunk& ch : g_chunks) {
        Vec3 d = ch.center - g_camPos;
        float dist2 = d.x * d.x + d.z * d.z;
        if (dist2 > 780.0f * 780.0f) continue;
        if (dist2 > ch.radius * ch.radius && dot(d, viewFwd) < -ch.radius) continue;
        glCallList(ch.solids);
    }

    // 5. Render Player WRC Rally1 Car
    if (g_cameraMode != 2) drawCar(g_car);

    // 6. Render Skidmarks, Tire Spray & Weather Particles
    renderSkidsAndParticles();

    // 7. Calculate Sun Lens Flare in screen space before clearing 3D matrices
    bool haveSunFlare = false;
    float sunScreenX = 0, sunScreenY = 0, sunDot = 0;
    if (cp.sunGlow > 0.0f) {
        Vec3 sdir = Vec3(cp.sunDir[0], cp.sunDir[1], cp.sunDir[2]).normalized();
        Vec3 vdir = (g_camTarget - g_camPos).normalized();
        sunDot = dot(vdir, sdir);
        if (sunDot > 0.45f) {
            double mv[16], pj[16];
            GLint vp[4];
            glGetDoublev(GL_MODELVIEW_MATRIX, mv);
            glGetDoublev(GL_PROJECTION_MATRIX, pj);
            glGetIntegerv(GL_VIEWPORT, vp);
            double wx = 0, wy = 0, wz = 0;
            if (gluProject(g_camPos.x + sdir.x * 1200.0,
                           g_camPos.y + sdir.y * 1200.0,
                           g_camPos.z + sdir.z * 1200.0,
                           mv, pj, vp, &wx, &wy, &wz) == GL_TRUE) {
                if (wz > 0.0 && wz < 1.0) {
                    haveSunFlare = true;
                    sunScreenX = (float)wx;
                    sunScreenY = (float)(g_winH - wy);
                }
            }
        }
    }

    // 8. Render 2D Telemetry, Pacenotes HUD & Cinematic Post-Processing
    renderHUD(haveSunFlare, sunScreenX, sunScreenY, sunDot);
}

// ============================================================================
//  X11 WINDOW + 4x MSAA GLX CONTEXT & MAIN EVENT LOOP
// ============================================================================
static XVisualInfo* chooseVisualWithMSAA(Display* dpy, int screen) {
    int attribsMSAA[] = {
        GLX_RGBA, GLX_DOUBLEBUFFER,
        GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, GLX_ALPHA_SIZE, 8,
        GLX_DEPTH_SIZE, 24,
        GLX_SAMPLE_BUFFERS, 1,
        GLX_SAMPLES, 4,
        None
    };
    if (XVisualInfo* vi = glXChooseVisual(dpy, screen, attribsMSAA)) return vi;
    int attribsFallback[] = {
        GLX_RGBA, GLX_DOUBLEBUFFER,
        GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8,
        GLX_DEPTH_SIZE, 24,
        None
    };
    return glXChooseVisual(dpy, screen, attribsFallback);
}

static void startDriveFromCanvas() {
    std::vector<Vec2> pts = g_rawPts;
    bool closed = g_closed;
    if (pts.size() < 4) {
        pts = defaultCanvasPoints();
        closed = true;
    }
    if (!generateStageFromCanvas(pts, closed)) {
        pts = defaultCanvasPoints();
        closed = true;
        generateStageFromCanvas(pts, closed);
    }
    buildWorldLists();
    resetStageSession();
    g_appState = STATE_RALLY_DRIVE;
}

int main(int /*argc*/, char** /*argv*/) {
    std::srand((unsigned)std::time(nullptr));
    g_rng.seed((uint32_t)std::time(nullptr));
    initFont();
    buildRockMeshes();

    g_dpy = XOpenDisplay(nullptr);
    if (!g_dpy) {
        std::fprintf(stderr, "HATA: X11 Display acilamadi!\n");
        return 1;
    }

    int screen = DefaultScreen(g_dpy);
    XVisualInfo* vi = chooseVisualWithMSAA(g_dpy, screen);
    if (!vi) {
        std::fprintf(stderr, "HATA: GLX Visual bulunamadi!\n");
        XCloseDisplay(g_dpy);
        return 1;
    }

    Colormap cmap = XCreateColormap(g_dpy, RootWindow(g_dpy, vi->screen), vi->visual, AllocNone);
    XSetWindowAttributes swa{};
    swa.colormap = cmap;
    swa.border_pixel = 0;
    swa.event_mask = ExposureMask | KeyPressMask | KeyReleaseMask |
                     ButtonPressMask | ButtonReleaseMask | PointerMotionMask |
                     StructureNotifyMask;

    g_win = XCreateWindow(g_dpy, RootWindow(g_dpy, vi->screen),
                          0, 0, g_winW, g_winH, 0, vi->depth, InputOutput, vi->visual,
                          CWBorderPixel | CWColormap | CWEventMask, &swa);
    XStoreName(g_dpy, g_win, "WRC Rally1 Hybrid Simulation [C++17 / OpenGL / 4x MSAA / X11]");
    Atom wmDelete = XInternAtom(g_dpy, "WM_DELETE_WINDOW", True);
    XSetWMProtocols(g_dpy, g_win, &wmDelete, 1);
    XMapWindow(g_dpy, g_win);

    g_ctx = glXCreateContext(g_dpy, vi, nullptr, GL_TRUE);
    glXMakeCurrent(g_dpy, g_win, g_ctx);
    XFree(vi);

    glEnable(GL_MULTISAMPLE);
    glEnable(GL_NORMALIZE);
    glShadeModel(GL_SMOOTH);

    initTextures();
    buildCarLists();
    initJoystick();

    auto prevTime = std::chrono::steady_clock::now();

    while (g_running) {
        while (XPending(g_dpy) > 0) {
            XEvent ev;
            XNextEvent(g_dpy, &ev);
            if (ev.type == ClientMessage) {
                if ((Atom)ev.xclient.data.l[0] == wmDelete) g_running = false;
            } else if (ev.type == ConfigureNotify) {
                g_winW = std::max(320, ev.xconfigure.width);
                g_winH = std::max(240, ev.xconfigure.height);
            } else if (ev.type == KeyPress) {
                KeySym ks = XLookupKeysym(&ev.xkey, 0);
                if (ks < 65536) g_keys[ks] = true;
                if (ks == XK_Escape) {
                    g_running = false;
                } else if (g_appState == STATE_DRAW_MAP) {
                    if (ks == XK_l || ks == XK_L) {
                        if (g_rawPts.size() >= 3) { g_closed = !g_closed; g_previewDirty = true; }
                    } else if (ks == XK_c || ks == XK_C) {
                        g_rawPts.clear(); g_closed = false; g_previewDirty = true;
                    } else if (ks == XK_Return || ks == XK_KP_Enter) {
                        startDriveFromCanvas();
                    }
                } else if (g_appState == STATE_RALLY_DRIVE) {
                    if (ks == XK_m || ks == XK_M) {
                        g_appState = STATE_DRAW_MAP;
                    } else if (ks == XK_k || ks == XK_K) {
                        selectClimate(true);
                        populateScenery();
                        buildWorldLists();
                        initWeather();
                    } else if (ks == XK_c || ks == XK_C) {
                        g_cameraMode = (g_cameraMode + 1) % 4;
                    } else if (ks == XK_v || ks == XK_V) {
                        g_car.model = (g_car.model + 1) % 3;
                    } else if (ks == XK_p || ks == XK_P) {
                        g_autopilot = !g_autopilot;
                    } else if (ks == XK_r || ks == XK_R) {
                        if (g_finished) resetStageSession();
                        else resetCarToNode(g_car.nearestNode);
                    }
                }
            } else if (ev.type == KeyRelease) {
                if (XEventsQueued(g_dpy, QueuedAfterReading)) {
                    XEvent nextEv;
                    XPeekEvent(g_dpy, &nextEv);
                    if (nextEv.type == KeyPress && nextEv.xkey.time == ev.xkey.time &&
                        nextEv.xkey.keycode == ev.xkey.keycode) continue;
                }
                KeySym ks = XLookupKeysym(&ev.xkey, 0);
                if (ks < 65536) g_keys[ks] = false;
            } else if (ev.type == ButtonPress) {
                if (g_appState == STATE_DRAW_MAP) {
                    if (ev.xbutton.button == Button1) {
                        g_drawing = true;
                        Vec2 p((float)ev.xbutton.x, (float)ev.xbutton.y);
                        if (g_rawPts.empty() || len2(p - g_rawPts.back()) > 6.0f) {
                            g_rawPts.push_back(p);
                            g_previewDirty = true;
                        }
                    } else if (ev.xbutton.button == Button3) {
                        g_rawPts.clear(); g_closed = false; g_previewDirty = true;
                    }
                }
            } else if (ev.type == ButtonRelease) {
                if (ev.xbutton.button == Button1) g_drawing = false;
            } else if (ev.type == MotionNotify) {
                g_mouse = Vec2((float)ev.xmotion.x, (float)ev.xmotion.y);
                if (g_appState == STATE_DRAW_MAP && g_drawing) {
                    if (g_rawPts.empty() || len2(g_mouse - g_rawPts.back()) > 8.0f) {
                        g_rawPts.push_back(g_mouse);
                        g_previewDirty = true;
                    }
                }
            }
        }

        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - prevTime).count();
        prevTime = now;

        if (g_appState == STATE_DRAW_MAP) {
            renderStateDrawMap();
        } else {
            updateSimulation(dt);
            renderStateRallyDrive();
        }
        glXSwapBuffers(g_dpy, g_win);
    }

    if (g_joy.fd >= 0) close(g_joy.fd);
    glXMakeCurrent(g_dpy, None, nullptr);
    glXDestroyContext(g_dpy, g_ctx);
    XDestroyWindow(g_dpy, g_win);
    XCloseDisplay(g_dpy);
    return 0;
}
