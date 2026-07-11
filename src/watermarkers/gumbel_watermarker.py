"""
This file defines the class for a watermarker that uses a Gumbling sampling technique to detect
watermarkers. Check the class docstring for more info.
"""
import torch
from src.watermarkers.watermarker import Watermarker

class GumbelWatermarker(Watermarker):
    """
    The GumbelWatermarker takes advantage of the Gumbel sampling technique to produce random
    variables that are highly correlated with the chosen tokens can be detected later on.
    
    # TODO later.
    """
