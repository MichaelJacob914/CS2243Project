# Narration script: "When does a diffusion policy make up its mind?"

For a later voiceover of `report/assets/video/critical_windows_explainer.mp4` (5 min 33 s, 10 scenes). Each section matches one manim scene in `scripts/exp14/video/explainer.py`. The on-screen captions say the same things in shorter form; the narration can run slightly ahead of them.

## 1. The question (0:00–0:31, S01Question)
A diffusion policy controls a robot one chunk of actions at a time. Each chunk starts as random noise, and the policy removes that noise over twenty denoising steps, numbered zero to nineteen.
DSRL trains a reinforcement-learning agent that steers the policy by choosing the noise it starts from, at step zero. Our variant lets the policy run its own denoising up to a later step k and swaps in the agent's noise there. The adaptive version picks k anew for every chunk.
That only pays off if the best step to steer at changes from chunk to chunk. Image diffusion models are known to commit to coarse choices inside a narrow band of steps, a critical window. If our policy has such windows and they move around, picking k per chunk makes sense. If they sit in one place, one fixed k is enough.

## 2. A toy with a known choice (0:31–0:56, S02Toy)
We start with a toy where we know the answer. A point moves from the left to the star and must go above or below a disc. A small diffusion policy, trained on demonstrations of both routes, outputs the next eight moves as one chunk.
Watch one chunk being made: at step zero it is pure noise, and each step removes some of it. Somewhere between step zero and step nineteen the policy decides: above or below. The question is where.

## 3. The measurement (0:56–1:55, S03Measure)
Take a finished chunk, here one that goes above. Add noise back to it, up to some step k, so it is partly undone. Then let the policy denoise it again, thirty-two times.
If the redraws disagree, the route was still open at step k. If they all agree, it was already decided. Noise back to step zero, and the redraws split between the routes. Noise back to a late step, and they all keep the original route.
Now do this for many chunks and ask how the variation in route splits up: differences between chunks, against differences among the redraws of one chunk. The share that lies between chunks is the resolved fraction. Zero means nothing is decided yet; one means the route is fully decided. Sweeping k from zero to nineteen traces out a curve. For this policy, the route goes from ten to ninety percent decided between steps two and ten.

## 4. The control (1:55–2:38, S04Control)
But the resolved fraction rises from zero to one for any distribution, so a rise alone proves nothing. We compare with a control: a Gaussian with the same mean and spread as the policy's chunks, run through exactly the same sampler code. It has no two routes to choose between.
Its redraws fan out and settle gradually, and its curve rises slowly. A critical window is a rise clearly steeper than the control's: about eight steps from ten to ninety percent for the policy, against about fifteen for the control.
Later on, past the disc, no route is left to choose, and the policy and the control should look alike. They do. This toy check passed in eight of nine repeats with fresh random seeds, all before any robot data were collected.

## 5. The robot task (2:38–2:58, S05Transport)
Now the robot: the frozen diffusion policy for robomimic transport, where two arms pass an object between bins. We measured 319 states from demonstrations and rollouts, labelled by task phase. At each state we tracked four properties of a chunk: which group of actions it belongs to, where actions form two or three groups; whether each gripper is open or closed; and a trained critic's score. Each got a resolved-fraction curve for the policy and for its Gaussian control, from 128 chunks with 16 redraws each. The tests and their thresholds were fixed before the data came in.

## 6. Finding 1: windows exist for discrete choices (2:58–3:37, S06Finding1)
Each dot is one state: the width of the policy's rise against the control's, measured on the log-signal-to-noise scale. Below the diagonal, the policy decides faster. The action group has a clear window at 44 of 77 states; the grippers at 60 and 71 of 319. Run the same test with a second Gaussian in place of the policy and almost nothing passes, so these are not false positives. The critic's value has no window: it settles as gradually as its control.

## 7. Finding 2: the window does not move (3:37–4:06, S07Finding2)
For each state, read off the step where a property is half decided, and group by task phase. The discrete choices sit near step five to six in every phase; the critic's value settles near step sixteen to seventeen, also in every phase. The largest gap between phases is under one step for the action group and gripper one. Gripper zero shows a larger gap, but it rests on three states in one phase and misses the significance bar. So choosing the step by task phase has nothing to gain. States within a phase still vary, which is why the simulator test comes next.

## 8. Finding 3: what is still undecided tracks the RL gain (4:06–4:32, S08Finding3)
Steering at step k can only change what is still undecided at k. The share of the critic's value still open, and the gain from RL training at each fixed step, rank the eight trained steps almost identically, with a rank correlation of 0.95. But the agreement comes from the late steps: from step fourteen on, little is left to change and the RL gain drops. Among steps zero to eleven the gain barely moves, so this explains why late steering fails but does not pick a best early step.

## 9. Finding 4: one chunk does not change success (4:32–5:00, S09Finding4)
So far we trusted the critic. Now real outcomes: at twenty states we regenerate only the first chunk from one of five steps, then let the policy finish the episode, 25,600 episodes in all. Success stays at 0.47 to 0.49, and which chunk was drawn decides almost none of it. The critic's value of the same chunks says the chunk decides a lot. The critic tells easy states from hard ones well, with an AUC of 0.86, but ranks chunks within a state at 0.57, barely above chance.

## 10. Recommendation and limit (5:00–5:33, S10Recommend)
Steer at one fixed early step, around k equals three to eight. Drop the learned per-chunk choice of step: the window does not move, and the critic cannot tell an adaptive agent which step to use. Longer RL runs agree: fixed k of three or five reach 0.91 to 0.92 success, against 0.87 for adaptive k, run before its entropy fix, and 0.82 at step zero.
The main limit: the simulator test changed only the first chunk, while an RL agent steers every chunk, and small effects could add up. The direct test, letting the critic pick every chunk at a fixed step, has not been run.
