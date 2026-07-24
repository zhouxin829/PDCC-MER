# best_config.py
# -*- coding: utf-8 -*-

BEST_CONFIGS = {
    "SIMS": {
        # TPLR
        "tplr_lambda": 0.3,
        "tplr_steps": 100,
        "tplr_sigma": 0.01,

        # PCRP
        "pcrp_steps": 4,
        "pcrp_strength": 0.5,
        "pcrp_temperature": 0.07,

        # RCCR
        "rccr_tau": 0.5,
        "rccr_lambda": 0.2,
        "top_k": 2,

        # Model structure
        "transformer_layers": 5,
        "nhead": 4,
        "out_dropout": 0.4,
    },

    "SIMS-v2": {
        # TPLR
        "tplr_lambda": 0.5,
        "tplr_steps": 200,
        "tplr_sigma": 0.02,

        # PCRP
        "pcrp_steps": 3,
        "pcrp_strength": 0.5,
        "pcrp_temperature": 0.1,

        # RCCR
        "rccr_tau": 3.0,
        "rccr_lambda": 0.1,
        "top_k": 2,

        # Model structure
        "transformer_layers": 4,
        "nhead": 2,
        "out_dropout": 0.3,
    },

    "MOSI": {
        # TPLR
        "tplr_lambda": 0.7,
        "tplr_steps": 350,
        "tplr_sigma": 0.05,

        # PCRP
        "pcrp_steps": 2,
        "pcrp_strength": 0.5,
        "pcrp_temperature": 0.07,

        # RCCR
        "rccr_tau": 0.5,
        "rccr_lambda": 0.3,
        "top_k": 2,

        # Model structure
        "transformer_layers": 2,
        "nhead": 4,
        "out_dropout": 0.5,
    },

    "MOSEI": {
        # TPLR
        "tplr_lambda": 0.05,
        "tplr_steps": 100,
        "tplr_sigma": 0.02,

        # PCRP
        "pcrp_steps": 3,
        "pcrp_strength": 0.2,
        "pcrp_temperature": 0.07,

        # RCCR
        "rccr_tau": 3.0,
        "rccr_lambda": 0.05,
        "top_k": 2,

        # Model structure
        "transformer_layers": 2,
        "nhead": 4,
        "out_dropout": 0.0,
    }
}