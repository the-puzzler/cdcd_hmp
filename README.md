# Continuous Diffusion on Catagorical Data
My implementation of Deep Mind's paper applied to an OTU (bacterial) dataset.
 
There are slight differences in implementation. Given the set based nature of OTUs, there is no ROPE use as in the original paper. There is no positional encoding at all. There may be some simplifications in certain places.
wandb: https://wandb.ai/matteopeluso1922/diffusion-hmp?nw=nwusermatteopeluso1922

## AIM
To create a diffusion model in this discrete space. To then build a mechanistic interpretability model on top of this.
The dataset is quite small, this is an excersize in implementation. I hope to scale this up to a much large OTU dataset. Eventually, instead of simple labels for OTUs, more rich information could be used such as genetic infomration.

After this model is trained to some degree, a SAE can be used to learn informative patterns and BACDIVE data can be used to see whether the features recovered by the SAE are enriched for anything interesting. I have run similair experiments on a 'masked' approach (MLM except theres no positional embedding so a CLS token is used instead). In those cases some interesting features were recovered. e.g. a feature enriched for pathogenic bacteria in this supposedly healthy human samples... interesting.

## Dataset
The dataset comes from the human micro-biome project. This has samples of OTUs from different sites on healthy humans.

## Different implementations

Simplified V1 and V2 contain simplified implementations of the original paper, not only removing positional encoding, but adjusting scaling factors, fixed functions, self conditioning, time conditioned layer normalisation in the transformer etc.

V2 is more like the paper than V1 in that it adds back in more stringent following of scaling and fixed transformation but leaves out the afdore mentioned other parts.

more_complete is as it sounds. The only major difference is that it leaves out positional encoding (and of course does not do language modelling: autoregressive/masked prediction). I took much inspiration from Francesco215