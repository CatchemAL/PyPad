import math
from enum import Enum, IntEnum


class ObsPlanes(IntEnum):
    PLAYER_PAWN = 0
    PLAYER_ROOK = 1
    PLAYER_KNIGHT = 2
    PLAYER_BISHOP = 3
    PLAYER_QUEEN = 4
    PLAYER_KING = 5
    OPP_PAWN = 6
    OPP_ROOK = 7
    OPP_KNIGHT = 8
    OPP_BISHOP = 9
    OPP_QUEEN = 10
    OPP_KING = 11
    TURN = 12
    CAN_PLAYER_KINGSIDE = 13
    CAN_PLAYER_QUEENSIDE = 14
    CAN_OPP_KINGSIDE = 15
    CAN_OPP_QUEENSIDE = 16
    HALFMOVE_CLOCK = 17
    EN_PASSANT_SQ = 18
    IS_TWOFOLD = 19

    @classmethod
    def shape(self) -> tuple[int, int, int]:
        return 20, 8, 8

    @classmethod
    def get(cls, label: str) -> int:
        return cls.__members__[label.upper()].value


class ActionPlanes(IntEnum):
    QUEEN_N_1 = 0
    QUEEN_NE_1 = 1
    QUEEN_E_1 = 2
    QUEEN_SE_1 = 3
    QUEEN_S_1 = 4
    QUEEN_SW_1 = 5
    QUEEN_W_1 = 6
    QUEEN_NW_1 = 7
    QUEEN_N_2 = 8
    QUEEN_NE_2 = 9
    QUEEN_E_2 = 10
    QUEEN_SE_2 = 11
    QUEEN_S_2 = 12
    QUEEN_SW_2 = 13
    QUEEN_W_2 = 14
    QUEEN_NW_2 = 15
    QUEEN_N_3 = 16
    QUEEN_NE_3 = 17
    QUEEN_E_3 = 18
    QUEEN_SE_3 = 19
    QUEEN_S_3 = 20
    QUEEN_SW_3 = 21
    QUEEN_W_3 = 22
    QUEEN_NW_3 = 23
    QUEEN_N_4 = 24
    QUEEN_NE_4 = 25
    QUEEN_E_4 = 26
    QUEEN_SE_4 = 27
    QUEEN_S_4 = 28
    QUEEN_SW_4 = 29
    QUEEN_W_4 = 30
    QUEEN_NW_4 = 31
    QUEEN_N_5 = 32
    QUEEN_NE_5 = 33
    QUEEN_E_5 = 34
    QUEEN_SE_5 = 35
    QUEEN_S_5 = 36
    QUEEN_SW_5 = 37
    QUEEN_W_5 = 38
    QUEEN_NW_5 = 39
    QUEEN_N_6 = 40
    QUEEN_NE_6 = 41
    QUEEN_E_6 = 42
    QUEEN_SE_6 = 43
    QUEEN_S_6 = 44
    QUEEN_SW_6 = 45
    QUEEN_W_6 = 46
    QUEEN_NW_6 = 47
    QUEEN_N_7 = 48
    QUEEN_NE_7 = 49
    QUEEN_E_7 = 50
    QUEEN_SE_7 = 51
    QUEEN_S_7 = 52
    QUEEN_SW_7 = 53
    QUEEN_W_7 = 54
    QUEEN_NW_7 = 55
    KNIGHT_NNE = 56
    KNIGHT_ENE = 57
    KNIGHT_ESE = 58
    KNIGHT_SSE = 59
    KNIGHT_SSW = 60
    KNIGHT_WSW = 61
    KNIGHT_WNW = 62
    KNIGHT_NNW = 63
    PROMOTE_KNIGHT_NW = 64
    PROMOTE_KNIGHT_N = 65
    PROMOTE_KNIGHT_NE = 66
    PROMOTE_ROOK_NW = 67
    PROMOTE_ROOK_N = 68
    PROMOTE_ROOK_NE = 69
    PROMOTE_BISHOP_NW = 70
    PROMOTE_BISHOP_N = 71
    PROMOTE_BISHOP_NE = 72

    @classmethod
    def shape(cls) -> tuple[int, int, int]:
        return 73, 8, 8

    @classmethod
    def size(cls) -> tuple[int, int, int]:
        return math.prod(cls.shape())

    @classmethod
    def get(cls, label: str) -> int:
        return cls.__members__[label.upper()].value


class KeyGames(Enum):
    ALPHA_ZEROS_IMMORTAL_ZUGZWANG_SANS = [
        "Nf3",
        "Nf6",
        "c4",
        "b6",
        "d4",
        "e6",
        "g3",
        "Ba6",
        "Qc2",
        "c5",
        "d5",
        "exd5",
        "cxd5",
        "Bb7",
        "Bg2",
        "Nxd5",
        "O-O",
        "Nc6",
        "Rd1",
        "Be7",
        "Qf5",
        "Nf6",
        "e4",
        "g6",
        "Qf4",
        "O-O",
        "e5",
        "Nh5",
        "Qg4",
        "Re8",
        "Nc3",
        "Qb8",
        "Nd5",
        "Bf8",
        "Bf4",
        "Qc8",
        "h3",
        "Ne7",
        "Ne3",
        "Bc6",
        "Rd6",
        "Ng7",
        "Rf6",
        "Qb7",
        "Bh6",
        "Nd5",
        "Nxd5",
        "Bxd5",
        "Rd1",
        "Ne6",
        "Bxf8",
        "Rxf8",
        "Qh4",
        "Bc6",
        "Qh6",
        "Rae8",
        "Rd6",
        "Bxf3",
        "Bxf3",
        "Qa6",
        "h4",
        "Qa5",
        "Rd1",
        "c4",
        "Rd5",
        "Qe1+",
        "Kg2",
        "c3",
        "bxc3",
        "Qxc3",
        "h5",
        "Re7",
        "Bd1",
        "Qe1",
        "Bb3",
        "Rd8",
        "Rf3",
        "Qe4",
        "Qd2",
        "Qg4",
        "Bd1",
        "Qe4",
        "h6",
        "Nc7",
        "Rd6",
        "Ne6",
        "Bb3",
        "Qxe5",
        "Rd5",
        "Qh8",
        "Qb4",
        "Nc5",
        "Rxc5",
        "bxc5",
        "Qh4",
        "Rde8",
        "Rf6",
        "Rf8",
        "Qf4",
        "a5",
        "g4",
        "d5",
        "Bxd5",
        "Rd7",
        "Bc4",
        "a4",
        "g5",
        "a3",
        "Qf3",
        "Rc7",
        "Qxa3",
        "Qxf6",
        "gxf6",
        "Rfc8",
        "Qd3",
        "Rf8",
        "Qd6",
        "Rfc8",
        "a4",
    ]

    CATCHEM_ALPHA_ZEROS_FIRST_GAME = [
        "Na3",
        "d6",
        "d3",
        "Bg4",
        "Bf4",
        "g5",
        "Qd2",
        "Bg7",
        "b3",
        "c6",
        "h3",
        "Qb6",
        "Nb5",
        "Qe3",
        "O-O-O",
        "Qe5",
        "c3",
        "Bh5",
        "Rh2",
        "gxf4",
        "a4",
        "h6",
        "Qe3",
        "f5",
        "Nd4",
        "Kd7",
        "Qxe5",
        "Kc7",
        "e4",
        "Nf6",
        "Ndf3",
        "dxe5",
        "Ne2",
        "Bg4",
        "c4",
        "b5",
        "cxb5",
        "e6",
        "Kd2",
        "Bf8",
        "Nfd4",
        "Nh5",
        "Kc3",
        "Kb7",
        "Kb2",
        "Na6",
        "b6",
        "Ba3+",
        "Kb1",
        "Bc1",
        "f3",
        "Be3",
        "a5",
        "Rhb8",
        "Nc3",
        "Ng7",
        "Nxc6",
        "Bxb6",
        "Rd2",
        "Ne8",
        "Ka1",
        "Bxh3",
        "Rxh3",
        "Kc8",
        "exf5",
        "h5",
        "Nd8",
        "Bxd8",
        "Kb1",
        "Rb6",
        "Rd1",
        "Nb8",
        "Ka1",
        "h4",
        "fxe6",
        "Ng7",
        "Na2",
        "Rc6",
        "d4",
        "Kc7",
        "Rxh4",
        "Rc1+",
        "Nxc1",
        "Nc6",
        "Rh6",
        "Nxa5",
        "Kb1",
        "Kc6",
        "Rd2",
        "Bb6",
        "Rf6",
        "Bd8",
        "Rf5",
        "Bb6",
        "d5+",
        "Kc5",
        "Na2",
        "Nh5",
        "Re2",
        "Kxd5",
        "g3",
        "Rf8",
        "Rb2",
        "Rc8",
        "Nb4+",
        "Kd6",
        "e7",
        "Rc5",
        "Rf8",
        "Rd5",
        "Rg8",
        "Nxb3",
        "Rc8",
        "Bd8",
        "Bd3",
        "Nd2+",
        "Kc1",
        "Kd7",
        "e8=Q+",
        "Kxc8",
        "Qxh5",
        "Kc7",
        "Qf7+",
        "Be7",
        "Bc4",
        "Rd6",
        "Qg8",
        "Bd8",
        "Qd5",
        "Kd7",
        "Qe6+",
        "Rxe6",
        "Rc2",
        "Rh6",
        "Bd5",
        "Ke8",
        "Ba8",
        "Kf8",
        "gxf4",
        "Bb6",
        "Kd1",
        "Ke8",
        "Rc4",
        "Kd8",
        "Rc6",
        "e4",
        "Re6",
        "Bf2",
        "Rg6",
        "Kd7",
        "Bxe4",
        "Rh2",
        "Rg2",
        "Nxf3",
        "Rg3",
        "Ke6",
        "Bd5+",
        "Kd7",
        "Bb7",
        "Ne1",
        "Rg5",
        "Bb6",
        "Ra5",
        "Nf3",
        "Rd5+",
        "Ke7",
        "Ba8",
        "Kf8",
        "Rg5",
        "Ke7",
        "f5",
        "Kf7",
        "Kc1",
        "Rh7",
        "Be4",
        "Nh2",
        "Bh1",
        "Rh8",
        "Rg6",
        "Ng4",
        "Rh6",
        "Bf2",
        "Kb2",
        "Rf8",
        "Na2",
        "Bg3",
        "f6",
        "Rd8",
        "Bb7",
        "Nh2",
        "Kc1",
        "Rh8",
        "Kc2",
        "Bb8",
        "Be4",
        "Rh7",
        "Bf5",
        "a5",
        "Rg6",
        "Rh6",
        "Nc1",
        "Rh8",
        "Rh6",
        "Rf8",
        "Bd3",
        "Kg8",
        "Nb3",
        "Rxf6",
        "Rg6+",
        "Kh7",
        "Na1",
        "Rf8",
        "Kd2",
        "Re8",
        "Rd6+",
        "Re4",
        "Nb3",
        "Nf3+",
        "Kc1",
        "Nd2",
        "Kc2",
        "Bxd6",
        "Bb5",
        "Rb4",
        "Nd4",
        "Rxd4",
        "Be8",
        "Rd3",
        "Kc1",
        "Rh3",
        "Bf7",
        "Kh8",
        "Bh5",
        "Bh2",
        "Kc2",
        "Bb8",
        "Bf3",
        "Bh2",
        "Bg4",
        "Rh6",
        "Bd7",
        "Rh7",
        "Bf5",
        "Rg7",
        "Be6",
        "Rh7",
        "Bc8",
        "Ra7",
        "Kc1",
        "Bg1",
        "Kb2",
        "Kh7",
        "Ka2",
        "Bh2",
        "Bg4",
        "Be5",
        "Bf5+",
        "Kh6",
        "Bg6",
        "Bd4",
        "Bf5",
        "Be3",
        "Be6",
        "Bg5",
        "Bh3",
        "Kh5",
        "Bg2",
        "Ne4",
        "Kb1",
        "Nf2",
        "Bd5",
        "Ne4",
        "Ba2",
        "Rh7",
        "Bc4",
        "Nf2",
        "Ba2",
        "Nh1",
        "Bc4",
        "Bc1",
        "Bf7+",
        "Kh4",
        "Bh5",
        "Kg3",
        "Bg6",
        "Kh3",
        "Bf7",
        "Kg3",
        "Bh5",
        "Kh3",
        "Ka2",
        "Bh6",
        "Bg4+",
        "Kh2",
        "Bf5",
        "Bg5",
        "Be6",
        "Bh6",
        "Bh3",
        "Bd2",
        "Bg2",
        "Rg7",
        "Bd5",
        "Rb7",
        "Bxb7",
        "Kg3",
        "Bc8",
        "Bf4",
        "Kb1",
        "Kf2",
        "Bh3",
        "Bh2",
        "Kb2",
        "Bb8",
        "Kc3",
        "Ke2",
        "Bf5",
        "Ke1",
        "Kd4",
        "Kf1",
        "Bd3+",
        "Kf2",
        "Kc5",
        "Ba7+",
        "Kc6",
        "Bb8",
        "Kb7",
        "Be5",
        "Ba6",
        "Ke3",
        "Ka8",
        "Kd2",
        "Ka7",
        "Bb8+",
        "Ka8",
        "Ke3",
        "Kxb8",
        "Kd2",
        "Bd3",
        "Kd1",
        "Ka8",
        "Ng3",
        "Bc2+",
        "Ke2",
        "Bb3",
        "Ke3",
        "Ka7",
        "Ke4",
        "Kb7",
        "Ke3",
        "Bg8",
        "a4",
        "Kc6",
        "Ne4",
        "Bh7",
        "Nf6",
        "Bg8",
        "Kf2",
        "Kb6",
        "Nxg8",
        "Ka7",
        "Kg3",
        "Ka6",
        "Kg4",
        "Kb6",
        "a3",
        "Kb5",
        "Ne7",
        "Ka4",
        "Kh3",
        "Kb5",
        "Ng8",
        "Kb6",
        "Kg2",
        "Ka6",
        "Ne7",
        "Kb6",
        "Kh3",
        "Kb5",
        "Kg3",
        "Kb6",
        "Ng6",
        "Ka6",
        "Kh3",
        "Ka7",
        "Kh2",
        "Kb7",
        "Nf8",
        "Kb6",
        "Kg2",
        "Ka6",
        "Ng6",
        "Kb6",
        "Nh8",
        "Ka6",
        "Kf1",
        "Kb6",
        "Ke2",
        "Ka6",
        "Kf1",
        "Ka7",
        "Nf7",
        "Ka8",
        "Kg2",
        "Ka7",
        "Kg3",
        "Ka6",
        "Kh3",
        "Ka7",
        "Nd6",
        "Ka6",
        "Nb7",
        "Kb6",
        "Kg3",
        "Ka6",
        "Kh3",
        "Ka7",
        "Kg3",
        "Kb8",
        "Kh3",
        "Ka7",
        "Kg2",
        "Ka6",
        "Na5",
        "Kb6",
        "Kh1",
        "Ka6",
        "Kh2",
        "Ka7",
        "a2",
        "Ka8",
        "Kg2",
        "Kb8",
        "Nb7",
        "Kc8",
        "Kf2",
        "Kxb7",
        "Ke1",
        "Ka6",
        "Ke2",
        "Ka7",
        "Kd3",
        "Kb6",
        "a1=B",
    ]

    SCHOLARS_MATE_SANS = ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7#"]

    WHITE_EN_PASSANT_SANS = ["e4", "a6", "e5", "d5"]
    BLACK_EN_PASSANT_SANS = ["a3", "e5", "a4", "e4", "d4"]

    WHITE_NO_QUEENSIDE_SANS = ["a3", "e5", "Ra2", "d5", "Ra1", "f5"]
    WHITE_NO_KINGSIDE_SANS = ["Nf3", "e5", "Rg1", "d5", "Rh1", "f5", "Ng1"]
    WHITE_NO_CASTLING_SANS = ["e4", "e5", "Ke2", "d5", "Ke1", "f5"]
    BLACK_NO_QUEENSIDE_SANS = ["e4", "a6", "e5", "Ra7", "Bc4", "Ra8", "Bd3"]
    BLACK_NO_KINGSIDE_SANS = ["e4", "h6", "e5", "Rh7", "Bc4", "Rh8", "Bd3"]
    BLACK_NO_CASTLING_SANS = ["e4", "e5", "Bc4", "Ke7", "Nf3", "Ke8"]

    WHITE_CASLTED_BLACK_NOT_YET_SANS = ["e4", "e5", "Bc4", "Bc5", "Nf3", "Ne7", "O-O", "f6", "Ng5", "a6"]

    @classmethod
    def get(cls, label: str) -> list[str]:
        return cls.__members__[label.upper()].value
