To use an *ingrained* structure and simulation/experiment correspondence as a starting point for a FANTASTX optimization, first ensure that a <code>progress.txt</code> file exists in the directory (initial structure selection is based on the results recorded in this file).
With this file in place, run the following (ensuring that the portion of code that loads the experimental image is consistent across all files and matches the procedure set up in <code>run.py</code>)
```python
python prepare_fantastx_input.py 

```
The above will create a <code>fantastx_start</code> directory which contains all items necessary to begin a FANTASTX optimization run for a grain boundary system. (i.e. initial structure <code>ingrained.POSCAR.vasp</code>, experimental image <code>experiment.npy</code>, and image simulation configuration file <code>sim_params</code>). Both the <code>sim_params</code> and <code>experiment.npy</code> file will be fixed during FANTASTX optimization and passed to each instance of <code>bicrystal.simulate_image()</code> and <code>score_ssim()</code>, respectively. To see how the *ingrained* package can be used in context of FANTASTX optimization runs, refer to the example script given in <code>run_fantastx_step.py</code> 

