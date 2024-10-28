// Some definitions presupposed by pandoc's typst output.
#let blockquote(body) = [
  #set text( size: 0.92em )
  #block(inset: (left: 1.5em, top: 0.2em, bottom: 0.2em))[#body]
]

#let horizontalrule = [
  #line(start: (25%,0%), end: (75%,0%))
]

#let endnote(num, contents) = [
  #stack(dir: ltr, spacing: 3pt, super[#num], contents)
]

#show terms: it => {
  it.children
    .map(child => [
      #strong[#child.term]
      #block(inset: (left: 1.5em, top: -0.4em))[#child.description]
      ])
    .join()
}

// Some quarto-specific definitions.

#show raw.where(block: true): block.with(
    fill: luma(230), 
    width: 100%, 
    inset: 8pt, 
    radius: 2pt
  )

#let block_with_new_content(old_block, new_content) = {
  let d = (:)
  let fields = old_block.fields()
  fields.remove("body")
  if fields.at("below", default: none) != none {
    // TODO: this is a hack because below is a "synthesized element"
    // according to the experts in the typst discord...
    fields.below = fields.below.amount
  }
  return block.with(..fields)(new_content)
}

#let empty(v) = {
  if type(v) == "string" {
    // two dollar signs here because we're technically inside
    // a Pandoc template :grimace:
    v.matches(regex("^\\s*$")).at(0, default: none) != none
  } else if type(v) == "content" {
    if v.at("text", default: none) != none {
      return empty(v.text)
    }
    for child in v.at("children", default: ()) {
      if not empty(child) {
        return false
      }
    }
    return true
  }

}

#show figure: it => {
  if type(it.kind) != "string" {
    return it
  }
  let kind_match = it.kind.matches(regex("^quarto-callout-(.*)")).at(0, default: none)
  if kind_match == none {
    return it
  }
  let kind = kind_match.captures.at(0, default: "other")
  kind = upper(kind.first()) + kind.slice(1)
  // now we pull apart the callout and reassemble it with the crossref name and counter

  // when we cleanup pandoc's emitted code to avoid spaces this will have to change
  let old_callout = it.body.children.at(1).body.children.at(1)
  let old_title_block = old_callout.body.children.at(0)
  let old_title = old_title_block.body.body.children.at(2)

  // TODO use custom separator if available
  let new_title = if empty(old_title) {
    [#kind #it.counter.display()]
  } else {
    [#kind #it.counter.display(): #old_title]
  }

  let new_title_block = block_with_new_content(
    old_title_block, 
    block_with_new_content(
      old_title_block.body, 
      old_title_block.body.body.children.at(0) +
      old_title_block.body.body.children.at(1) +
      new_title))

  block_with_new_content(old_callout,
    new_title_block +
    old_callout.body.children.at(1))
}

#show ref: it => locate(loc => {
  let target = query(it.target, loc).first()
  if it.at("supplement", default: none) == none {
    it
    return
  }

  let sup = it.supplement.text.matches(regex("^45127368-afa1-446a-820f-fc64c546b2c5%(.*)")).at(0, default: none)
  if sup != none {
    let parent_id = sup.captures.first()
    let parent_figure = query(label(parent_id), loc).first()
    let parent_location = parent_figure.location()

    let counters = numbering(
      parent_figure.at("numbering"), 
      ..parent_figure.at("counter").at(parent_location))
      
    let subcounter = numbering(
      target.at("numbering"),
      ..target.at("counter").at(target.location()))
    
    // NOTE there's a nonbreaking space in the block below
    link(target.location(), [#parent_figure.at("supplement") #counters#subcounter])
  } else {
    it
  }
})

// 2023-10-09: #fa-icon("fa-info") is not working, so we'll eval "#fa-info()" instead
#let callout(body: [], title: "Callout", background_color: rgb("#dddddd"), icon: none, icon_color: black) = {
  block(
    breakable: false, 
    fill: background_color, 
    stroke: (paint: icon_color, thickness: 0.5pt, cap: "round"), 
    width: 100%, 
    radius: 2pt,
    block(
      inset: 1pt,
      width: 100%, 
      below: 0pt, 
      block(
        fill: background_color, 
        width: 100%, 
        inset: 8pt)[#text(icon_color, weight: 900)[#icon] #title]) +
      block(
        inset: 1pt, 
        width: 100%, 
        block(fill: white, width: 100%, inset: 8pt, body)))
}



#let article(
  title: none,
  authors: none,
  date: none,
  abstract: none,
  cols: 1,
  margin: (x: 1.25in, y: 1.25in),
  paper: "us-letter",
  lang: "en",
  region: "US",
  font: (),
  fontsize: 11pt,
  sectionnumbering: none,
  toc: false,
  toc_title: none,
  toc_depth: none,
  doc,
) = {
  set page(
    paper: paper,
    margin: margin,
    numbering: "1",
  )
  set par(justify: true)
  set text(lang: lang,
           region: region,
           font: font,
           size: fontsize)
  set heading(numbering: sectionnumbering)

  if title != none {
    align(center)[#block(inset: 2em)[
      #text(weight: "bold", size: 1.5em)[#title]
    ]]
  }

  if authors != none {
    let count = authors.len()
    let ncols = calc.min(count, 3)
    grid(
      columns: (1fr,) * ncols,
      row-gutter: 1.5em,
      ..authors.map(author =>
          align(center)[
            #author.name \
            #author.affiliation \
            #author.email
          ]
      )
    )
  }

  if date != none {
    align(center)[#block(inset: 1em)[
      #date
    ]]
  }

  if abstract != none {
    block(inset: 2em)[
    #text(weight: "semibold")[Abstract] #h(1em) #abstract
    ]
  }

  if toc {
    let title = if toc_title == none {
      auto
    } else {
      toc_title
    }
    block(above: 0em, below: 2em)[
    #outline(
      title: toc_title,
      depth: toc_depth
    );
    ]
  }

  if cols == 1 {
    doc
  } else {
    columns(cols, doc)
  }
}
#show: doc => article(
  toc_title: [Table of contents],
  toc_depth: 3,
  cols: 1,
  doc,
)


= Introduction
<introduction>
== Background
<background>
How does the brain efficiently process the overwhelming amount of sensory input that it receives from a constantly changing and uncertain environment? According to the predictive coding \(PC) framework, the brain’s key solution is to actively predict incoming sensory input based on past observations and to prioritize the processing of unpredicted sensory information #cite(<huang2011>);. PC envisions the brain as a prediction machine, where predictions are made through prior experience or expectation. These predictions are compared to the actual sensory input, and the difference, known as prediction error, is used to refine the internal model of the environment and orient attention towards unexpected features of the input. This allows the brain to minimize the prediction error and allocate its finite resources more efficiently #cite(<vinck2022>);.

Predictive coding is generally thought to be implemented by the cortex. Indeed, the mammalian cortex is organized hierarchically, with lower level areas processing more basic sensory features and higher areas integrating this information into more complex representations #cite(<felleman1991>);. Visual cortex is a prime example of this hierarchy, with primary visual cortex \(V1) processing simple visual features like contrast and edges, and higher visual areas\(e.g.~V2, V4) extracting more complex features of stimuli like texture and shape \[XSX\]. The hierarchy of cortical processing allows the brain to make predictions at multiple levels of abstraction and compare these predictions to the actual sensory input. according to PC the communication between hierarchical cortical areas occurs through feedforward \(FF) and feedback \(FB) connections. FB projections involve top-down propagation of prediction from higher to lower area while FF projections involve bottom up assembly of sensory input and prediction error from lower to higher areas to update the internal model. FF and FB connections have distinct laminar origins and targets within the cortex. FF connections mainly originate from layers 3 and 5 \(of lower areas) and target the layer 4 \(of higher area). while, FB connections arise from layers 2 and 6 \(of higher area) and target all layers except for layer 4 \(of lower area)#cite(<vinck2022>);.

Prominent studies in primates visual systems #cite(<vankerkoerle2014>) suggest neural oscillations might play an important role in communication between cortical regions through FF-FB connections. The studies found that FF propagation is associated with gamma-band oscillations, while FB propagation involves alpha-band \(8–12 Hz) oscillations. Similarly, a study on mice #cite(<aggarwal2022>) vision revealed FF waves in the 30–50 Hz range and FB waves in the 3-6 Hz range, where the phase of the FB oscillations modulated the amplitude of the FF oscillations. However, this was the only study on mice, to the best of my knowledge, that has investigated the FF-FB waves in the visual system. Additionally, the study utilized a simple visual stimulus that did not involve any prediction and higher-level cognitive processes.

The mouse visual system offers several advantages for studying predictive coding. The simpler and well-studied visual processing hierarchy in mice allows for more straightforward analysis of neural data. Genetic manipulation tools in mice also provide unique opportunities to explore cortical micro-circuitry and manipulate neural pathways, which is challenging in primates. Additionally, mouse studies are more cost-effective and logistically feasible compare to primates, making them an attractive model for large-scale explorations of complex brain functions like predictive coding.

The International Brain Laboratory \(IBL) #cite(<benson2023>) provides an extensive open-access dataset recorded from more than 100 mice trained to perform a perceptual decision-making task. In this task, mice are presented with a visual stimulus of controlled contrast and are required to move the stimulus to the center of the screen using a steering wheel. The stimulus appears on the right or left side of the screen, with a fixed probability for blocks of trials to create a predictable pattern. The transitions between these blocks are designed to be unpredictable.

== Current project
<current-project>
= Methods
<methods>
== Data source and recording details
<data-source-and-recording-details>
=== International Brain Laboratory \(IBL)
<international-brain-laboratory-ibl>
We used the open-access dataset from the International Brain Laboratory \(IBL). IBL provides a comprehensive set of recordings collected from more than 100 mice across 11 laboratories performing a standardized perceptual decision-making task. Data are collected from 267 brain regions by inserting 547 Neuropixel probes covering most of the left hemisphere and spanning the forebrain, midbrain and cerebellum, as well as the right hindbrain #cite(<benson2023>);.

=== Task detail
<task-detail>
In the IBL task \(Figure 1), head-fixed mice had to move a visual stimulus to the center by turning a wheel with their front paws. At the start of each trial, the mouse was required to refrain from moving the wheel for a quiescence period lasting between 400 and 700 milliseconds. After this period, a visual stimulus \(Gabor patch) appeared on either the left or right side of the screen accompanied by a 100-millisecond tone \(5 kHz sine wave). If the mouse correctly moved the stimulus to the center by turning the wheel over 35°, it received a 3 µL water reward. Incorrect responses or failing to respond within 60 seconds resulted in a 500-millisecond burst of white noise and a timeout #cite(<benson2023>);. As shown in Figure 1c, mice typically responded quickly within 2 seconds. The stimulus is always presented for the first 1 second regardless of response time \(RT). RT is defined as the time after stimulus when the wheel rotation exceeds the threshold.

The experiment began with 90 unbiased trials where the stimulus appeared equally on both sides. The stimulus contrast levels were presented in a ratio of \[2:2:2:2:1\] for contrasts \[100%, 25%, 12.5%, 6%, 0%\]. After this initial block, trials were organized into biased blocks where the likelihood of the stimulus appearing on one side was fixed at 20% for the left and 80% for the right in "right blocks" or vice versa in "left blocks." These blocks consisted of 20 to 100 trials determined by a truncated geometric distribution with stimulus contrast levels ratio identical to those in the unbiased block. In 0% contrast trials where no stimulus was visible, the side assignment followed the block bias \(e.g., right side for right blocks) #cite(<benson2023>);.

#figure([
#box(width: 550.9565217391304pt, image("images/task.png"))
], caption: figure.caption(
position: bottom, 
[
a) Example session block diagram and IBL task. Each block of consecutive trials after 90 trials varied the probability of the stimulus being on the right side. B) A timeline of the main events and variables of the IBL task. After a quiescence period, stimulus appears on screen alongside a go cue tone. Mice had to bring the stimulus to the center by turning the wheel. When the wheel rotation reaches the threshold 35 ° or after 60 s of no response, positive or negative feedback is provided depending on the mice choice. \(a) and \(b) are extracted from . C) Distribution of response time \(RT) with color blue and stimulus offset time with color yellow relative to stimulus onset. Note that there is always a stimulus presented for the first 1 second even though the mice typically answer sooner.~
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


=== Electrophysiological recording
<electrophysiological-recording>
The neural recordings were conducted using Neuropixel probes with 384 recording channels and 960 low-impedance sites on a single shank #cite(<benson2023>);. Neuropixel probes are advanced silicon-based neural recording devices designed for high-density recording of neural activity across large populations of neurons with precise spatial and temporal resolution #cite(<jun2017>);. After the recordings, electrode tracks were reconstructed by performing serial-section 2-photon microscopy. A region was then assigned to each recording site \(and inferred single neurons) within the Allen Common Coordinate Framework #cite(<benson2023>);.

== Preprocessing of electrophysiological data
<preprocessing-of-electrophysiological-data>
=== Exclusion of channels and trials
<exclusion-of-channels-and-trials>
Local field potential \(LFP) datasets alongside their corresponding behavioral data and channel locations were extracted for sessions that included at least one channel in the primary visual cortex. The destriping function of the IBL Python toolbox was applied as the first step of preprocessing to correct for the biases induced by the sequential acquisition of the raw voltage traces #cite(<unified2024>);. This was followed by downsampling from 1000 Hz to 500 Hz to decrease the size of the data files. Next, channels were excluded based on three criteria: \(i) those not located the primary visual cortex, \(ii) those displaying excessively high variance according to power spectral density, \(iii) and those with an excessively low coherence with neighboring channels.

We faced an unexpected problem with IBL LFP data due to amplifier saturation. Indeed, Neuropixel probes \(especially earlier version) have a limited dynamic range that was frequently exceeded during the task \(in particular, when the animal licked the spout to harvest water reward). For the analysis of spikes, this issue is less problematic as it only prevent from detecting spikes during the saturation. However, for the analysis of LFP, it introduces very salient artifacts and dramatically increase inter-trial variance in power, amplitude and phase estimates, potentially leading to erroneous conclusions. Therefore, we designed a custom exclusion procedure tailored to capture this specific problem. Trials were excluded based on the skewness of the absolute value of their first-order temporal derivative \(threshold set to 1.5). Indeed, high skewness values typically reflect the presence of sudden, large amplitude changes in an otherwise mostly flat signal. By excluding these trials, our analyses focus on more consistent and representative portions of the data, improving the reliability of the results. Unless specified otherwise, all remaining trials were included in the presented analyses \(i.e., missed, incorrect and slow responses).

=== Common average reference
<common-average-reference>
Unless otherwise specified, our electrophysiological analyses used a common average reference scheme. The common reference was recomputed as the mean of all channels of interest per animal \(i.e., those located in the primary visual cortex), after excluding noisy channels). This approach was chosen to limit the influence of electrical potentials outside of visual areas as well as the influence of non-physiological noise.

=== Current Source Density \(CSD)
<current-source-density-csd>
To remove the effects of volume conduction on the LFP data and improve spatial resolution, we used Current Source Density \(CSD) analysis. CSD is a technique that estimates the local current flow in the brain by calculating the second spatial derivative of the recorded potentials to reduce the influence of distant sources. First, the Euclidean distances between adjacent channels were computed using the channels’ location relative to the end of the probe \(axial) and their location relative to the middle of the probe \(lateral). The Euclidean distance between adjacent channels $i$ and $i plus.minus 1$ was calculated as:

$ d_(i , i plus.minus 1) = sqrt((x_(i plus.minus 1) - x_i)^2 + (y_(i plus.minus 1) - y_i)^2) $

Where $d_(i , i plus.minus 1)$ is the distance between channel $i$ and its adjacent channel $i + 1$ \(next channel) or $i - 1$ \(previous channel), $x_i$ and $x_(i plus.minus 1)$ are the axial coordinates of the channels, $y_i$ and $y_(i plus.minus 1)$ are the lateral coordinates.

Then the second spatial derivative of the LFP signals was computed as:

$ C S D_i = frac(V_(i + 1) - V_i, d_(i , i + 1)^2) - frac(V_i - V_(i - 1), d_(i , i)^2) $

Here $C S D_i$ represents the current source density at channel $i$, $V_i$ is the voltage at channel $i$, $V_(i + 1)$ and $V_(i - 1)$ are the voltages at the adjacent channels $i + 1$ and $i - 1$ respectively.

In one-dimensional CSD analysis, it is typically assumed that channels are spaced uniformly \(i.e., $d_(i i + 1) = d_(i i - 1)$). However, in this project, we accounted for non-uniform spacing to enhance accuracy and enable the removal of noisy channels without risking the spread of artifacts to adjacent channels. The python script for CSD computation can be found in the "CSD\_computation" Jupyter notebook in the GitHub repository.

== Time-Frequency power analysis
<time-frequency-power-analysis>
For the time-frequency analysis, we chose the multitaper method. This method is known to be well-suited for situations where specific frequency bands are not preselected and the goal is a broad exploration of all frequencies. Multitaper parameters were selected in a way where frequency resolution was prioritized slightly over temporal resolution, especially at lower frequencies. In this regard, power and phase were calculated using MNE’s multitaper function with the following parameters: a frequency range of 2-45 Hz with a step size of 0.5 Hz for the 2-10 Hz range and 1 Hz for frequencies above 10 Hz and a time-bandwidth product of 3.5 with the number of cycles at each frequency point set to half of the corresponding frequency \($upright("n-cycles") = f / 2$). These parameters were found to be optimal for our specific data and goals. A detailed comparison of different parameter settings can be found in the "#emph[tf\_resolution];" Jupyter notebook in the GitHub repository.

Baseline correction were applied to the time frequency data with baseline defined as the interval from -0.7 to -0.5 seconds relative to stimulus onset. For baseline correction, the percentage change method were used which can be expressed with the following formula:

$ upright("Corrected Power") (t , f) = (frac(P (t , f) - upright("Baseline") (f), upright("Baseline") (f))) times 100 $

Where $P (t , f)$ is the power at a specific time $(t)$ and frequency $(f)$, and $upright("Baseline") (f)$ is the averaged power within the baseline interval for each frequency.

== Inter trial Phase coherence \(ITC)
<inter-trial-phase-coherence-itc>
Inter-trial phase coherence \(ITC) was computed using MNE’s built-in function. ITC is a measure of the consistency of the phase of a signal across different trials at a given time and frequency. Mathematically, ITC is calculated as the magnitude of the average of normalized complex phase values across trials. For each trial, the phase of the signal, denoted as $phi.alt (f , t)$, is extracted at each frequency $f$ and time point $t$. These phase values are then represented as unit vectors on the complex plane, i.e., $e^(i phi.alt (f , t))$.

The ITC at a particular time-frequency point is then defined as:

$ upright("ITC") (f , t) = lr(|1 / N sum_(n = 1)^N e^(i phi.alt_n (f , t))|) $

where $N$ is the number of trials, and $phi.alt_n (f , t)$ is the phase at frequency $f$ and time $t$ for the $n$-th trial. The resulting ITC value ranges from 0 to 1, where 0 indicates no phase consistency across trials, and 1 indicates perfect phase alignment across all trials.

== Phase-Amplitude Coupling
<phase-amplitude-coupling>
Phase-Amplitude Coupling \(PAC) quantifies the interaction between the phase and amplitude of two distinct frequency bands, typically involving the phase of a low-frequency oscillation and the amplitude of a high-frequency oscillation. In this study, PAC was computed for phase frequencies ranging from 2 to 7 Hz and amplitude frequencies from 25 to 80 Hz using the TensorPAC Python module #cite(<combrisson2020>);. The process begins with the extraction of the instantaneous phase of the low-frequency signal and the amplitude envelope of the high-frequency signal carried out through Morlet wavelets. The interaction between these signals is then evaluated to determine how the phase of slower oscillations modulates the amplitude of faster oscillations. In this project, the Gaussian Copula \(GC) method was employed to compute PAC for a time window spanning 500 ms before stimulus onset to 1 second after the stimulus. Compared to other methods such as Phase Locking Value, GC is more robust to shifts in overall signal amplitude #cite(<combrisson2020>);.

The core of the GC method involves calculating the mutual information between normalized amplitude and phase to quantify the degree to which the phase of the low-frequency oscillation governs the amplitude of the high-frequency oscillation. This mutual information provides a lower-bound estimate of the PAC that is robust to overall amplitude shifts. Mathematically, this can be expressed as: $ g c P A C = I (a (t) ; sin (phi.alt (t)) , cos (phi.alt (t))) $

Where $I$ denotes the mutual information, $a (t)$ represents the normalized amplitude signal, and $phi.alt (t)$ represents the normalized phase signal.

After computing PAC, the values were normalized for each channel using z-score normalization, which involves subtracting the mean and dividing by the standard deviation. This process standardizes the PAC values and makes them comparable across channels and subjects. Following normalization, the values were averaged across all frequencies and for two distinct time windows: before the stimulus \(-0.5 to 0 seconds) and after the stimulus \(0 to 1 second).

== Statistical Analysis
<statistical-analysis>
=== Analysis of variance \(ANOVAs)
<analysis-of-variance-anovas>
=== Cluster-based statistics
<cluster-based-statistics>
=== multiple comparison for time frequency ITC estimate
<multiple-comparison-for-time-frequency-itc-estimate>
= Results
<results>
== Data summary
<data-summary>
A total of 63 probes were identified in the IBL datasets, with at least one channel assigned to the primary visual cortex \(V1) \(see fix X a;b for one insertion example). From the initial dataset, 7 and 15 insertions were excluded due to over 40% noisy channels and trials, respectively. In the end, 41 insertions were retained, consisting of 2,262 total channels and 25,075 trials. On average, each probe was associated with 54.83 channels in V1 \(range: 2 to 118), with an average of 532.66 trials per session \(range: 276 to 1,098) \(see fig X d ) . Among the total number of channels, 212 \(9.37%) were in layer 1, 456 \(20.16%) in layer 2/3, 338 \(14.94%) in layer 4, 650 \(28.74%) in layer 5, and 606 \(26.79%) in layer 6 \(see fig X c).

#figure([
#box(width: 742.6354515050167pt, image("images/summary.png"))
], caption: figure.caption(
position: bottom, 
[
#strong[a)] Coronal slice of the Allen Brain Atlas, highlighting the layers of the primary visual cortex \(V1) with distinct colors: yellow for layer 1, light blue for layers 2/3, red for layer 4, blue for layer 5, and green for layer 6. The image base is extracted from the Allen Brain Atlas \(#link("https://atlas.internationalbrainlab.org");) at an Anterior-Posterior \(AP) coordinate of -3140 µm . #strong[b)] Coronal slice from the Allen Brain Atlas that shows an example of a probe insertion site in a mouse brain \(subject name: NYU-12). The black line represents the probe path, starting in V1 and ending in the midbrain reticular nucleus \(approximately). The image is taken from the IBL online data visualization tool \(#link("https://viz.internationalbrainlab.org");). #strong[c)] Pie chart illustrating the proportional distribution of each V1 layer, using the same color scheme as in panel \(a). #strong[d)] Scatter plot showing the number of trials \(range: 276-1098) and channels \(range: 2-118) for the included sessions. The mean number of channels \(54.83) is indicated by a dashed blue vertical line, and the mean number of trials \(532.66) is represented by a dashed red horizontal line.
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


== Behavioral results 
<behavioral-results>
In line with previous results on whole sessions, mice performed correctly on 80.7% ± 5.8% \(mean ± s.d.) of the trials with reaction time \(RT) of 1.73 ± 5.7 seconds \(mean ± s.d.). RT is defined as the time interval between stimulus onset and when wheel rotation reach threshold of 35° ; and performance is computed as a percent of correct trials over total number of trials. As illustrated in \(fig X a;b), Performance improved and reaction times decreased on trials with higher stimulus contrast. In 0% contrast trials, where mice had to rely only on their expectation and prior experience, they made correct choices in 57% ± 8% \(mean ± s.d.).

#figure([
#box(width: 624.6421404682275pt, image("images/behavioral.png"))
], caption: figure.caption(
position: bottom, 
[
Behavior results: #strong[a)] illustration of reaction time for different stimulus contrast level using boxplot. #strong[b)] Illustration of performance \(as a percent of correct trials over total number of trials) for each contrast levels using boxplots.
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


== Inter trial phase coherence \(ITC)
<inter-trial-phase-coherence-itc-1>
The ITC analysis indicated significant phase alignment in the low-frequency range \(2-8 Hz) within the \[0, 0.5\] second interval following the stimulus \(see Fig. X a). To ensure that these findings were not due to chance and to correct for multiple comparisons, we applied the MNE one-sample cluster permutation test \(refer to the Statistical Analysis section of the Methods for more details ). The significant clusters, marked by the black line in Fig. X a , demonstrate that the low-frequency ITC during the 0-0.5 second period was statistically meaningful, with a p-value of 0.001. Additionally, as illustrated in Fig. X c, there were no significant differences in ITC across the V1 layers in the low-frequency range.

To assess whether the observed ITC levels were influenced by the stimulus, ITC average levels were compared for each level of stimulus contrast. The average ITC was computed for the low-frequency range \(2-8 Hz) and within the 0-0.5 second time window post-stimulus. As illustrated in Fig. X b, an increase in stimulus contrast generally resulted in a higher mean ITC. Interestingly, the only group of trials that did not support this trend was trials without stimulus \(i.e.~contrast 0%), which will be discussed in the next section.

To statistically evaluate whether the mean ITC was significantly affected by contrast levels, further analysis was undertaken. Given that the Shapiro-Wilk normality test did not confirm normality in the data distribution, the Friedman test, a non-parametric alternative to repeated measures ANOVA, was employed. The results of the Friedman test indicated a highly significant effect of contrast level on ITC mean, with a test statistic of 77.98 and a p-value of 4.66\*10#super[-16];, indicating that variations in ITC across different contrast levels were unlikely to have occurred by chance. Due to the significant effect of contrast level on ITC mean identified by the Friedman test, a post-hoc Nemenyi test was performed to determine which specific contrast levels contributed to the observed differences. The Nemenyi test was chosen as it is appropriate for pairwise comparisons following a Friedman test. The post-hoc analysis results are presented in fig #underline[X d,] with p-values indicating the significance of differences between each pair of contrast levels.

#figure([
#box(width: 722.4080267558528pt, image("images/ITC.png"))
], caption: figure.caption(
position: bottom, 
[
Inter trial phase coherence \(ITC). #strong[a)] ITC average across all subjects and cortical layers relative to stimulus onset. Significant clusters \(p \= 0.001) are indicated by black lines, as determined by the MNE one-sample cluster permutation test.#strong[b)] Comparison of ITC averages across different stimulus contrast levels using a box plot. The ITC average was computed for the 2-8 Hz frequency range within the 0-0.5 second time window post-stimulus, aggregated across all subjects and layers. #strong[c)] Low-frequency ITC averages for each V1 layer relative to stimulus onset. The layer-specific averages are depicted with solid lines, and their 95% confidence intervals are shaded around the lines in distinct colors: yellow for layer 1, light blue for layers 2/3, red for layer 4, blue for layer 5, and green for layer 6. #strong[d)] Nemenyi post-hoc test p-value results for each pair of contrast levels. Lower p-values indicate significant differences between the ITC average distributions for each pair, represented by a heatmap ranging from yellow \(low p-values) to black \(high p-values).
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


== Time frequency analysis results
<time-frequency-analysis-results>
Although there was substantial variability across subjects, the time-frequency analysis of V1 revealed two notable oscillations in relation to the visual stimulus: first, an increase in high-frequency power within the gamma band range \(20–40 Hz), and second, a concurrent decrease in lower-frequency power within the 2–7 Hz range. The gamma increase was more transient, while the lower frequency inhibition persisted for a longer duration \(see Figure X). As shown in Figure Xa,b, the observed frequency band changes exhibited a similar pattern across the different layers of V1. Statistical tests were not applied to quantitatively evaluate the layer-specificity of these effects, as the variability in the number of channels across layers and subjects did not allow for such an analysis.

The averaged power of each frequency band over the first 1 second after the stimulus was compared for trials with different contrast levels. The Friedman test revealed a statistically significant effect of contrast level on power modulation in both the low-frequency band \(test statistic: 20.54, p-value: 0.00039) and the gamma band \(test statistic: 18.88, p-value: 0.00083), indicating that power in these bands was significantly influenced by the stimulus contrast. However, as you can see in Figure Xa,b, the frequency band average power across different contrast levels is not quite observable. Additionally, the post-hoc test results shown in Figure Xc,d indicate that the comparisons were mainly significant only in comparison to the 100% contrast condition.

This not readily observable difference is believed to be mainly due to inter-subject variability, with some subjects showing significant contrast-related modulations, while others do not. In general, especially in sessions with high effects of contrast on power bands, higher contrast stimuli involved greater increases in gamma and decreases in theta band power.

#figure([
#box(width: 440.9096989966555pt, image("images/TF.png"))
], caption: figure.caption(
position: bottom, 
[
#strong[Average time frequency representation.] Time-frequency is averaged over all channels in the primary visual cortex, computed using the multitaper method. The data is baseline-corrected using the -0.7 to -0.5 s pre-stimulus interval, with time relative to stimulus onset \(marked by dashed black vertical line). Power is shown in percent units with a blue-to-warm colormap.
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


#figure([
#box(width: 722.4080267558528pt, image("images/layer_power.png"))
], caption: figure.caption(
position: bottom, 
[
#strong[Average frequency band power across time for all layers of V1];. #strong[a)] low frequency \(2-7 Hz) average power for each V1 layer relative to stimulus onset \(marked by dashed black vertical line). The layer-specific averages are depicted with solid lines, and their 95% confidence intervals are shaded around the lines in distinct colors: yellow for layer 1, light blue for layers 2/3, red for layer 4, blue for layer 5, and green for layer 6. #strong[b)] similar to panel \(a) but for higher frequency band \(20-40 Hz)
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


#figure([
#box(width: 664.3745819397993pt, image("images/contrast_power.png"))
], caption: figure.caption(
position: bottom, 
[
Affects of stimulus contrast on low and high frequency average power. #strong[a)] The box plots represents the distribution of low-frequency \(2-7 Hz) average power across different contrast levels.The power is averaged over the first 1 second after stimulus across all channels of each subject \(N\= 41). The central line in each box indicates the median power average value, while the box edges represent the inter-quartile range \(IQR) and whiskers extend to 1.5 times the IQR. #strong[B)] similar to panel \(b) but for high frequency range \(20-40 hz). #strong[C)] Nemenyi post-hoc test p-value results comparing low frequency average power for pairwise contrasts levels. Lower p-values indicate significant differences between the low frequency average power distributions for each pair, represented by a heatmap ranging from yellow \(low p-values) to black \(high p-values). #strong[d)] similar to panel \(c) but for high frequency \(20-40 Hz) average power
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


== Phase amplitude coupling \(PAC)
<phase-amplitude-coupling-pac>
Phase-amplitude coupling \(PAC) analysis revealed that the phase of low-frequency oscillations \(2-7 Hz) modulates the amplitude of high-frequency oscillations \(25-80 Hz). In Figure X, an example from a single channel illustrates how the amplitude of high-frequency oscillations changes depending on the phase of low-frequency oscillations during the period after the stimulus. This example visually demonstrates the coupling effect; however, it is important to note that this figure serves solely as an illustration of the amplitude modulation by phase, and no specific coupling method \(e.g., Gaussian copula) was applied to quantify this effect.

#figure([
#box(width: 1156.3636363636363pt, image("images/bin_amplitude.png"))
], caption: figure.caption(
position: bottom, 
[
#strong[Binned Amplitude by Phase for a channel in V1];. This figure illustrates the binned amplitude of high-frequency oscillations \(25-80 Hz) as a function of the phase of low-frequency oscillations \(2-7 HZ) for four time windows. From left to right : \(-1, -0.5), \(-0.5, 0), \(0, 1), and \(1, 2) with time in seconds relative to the stimulus onset. The data were binned into 20 equal-sized phase bins, and the amplitude was averaged within each bin. The non-uniform distribution of amplitudes indicate high possibility of coupling. As you can see, the time period one second after stimulus \(0, 1s) shows observable non-uniform distribution of amplitudes with the highest value for phase zero. This plot is intended to illustrate the relationship between phase and amplitude without applying any coupling quantification. The plot is made using Tensorpac python module #cite(<combrisson2020>)
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


Comparing PAC values for the time periods before and after the stimulus revealed a significant increase in PAC values following the stimulus. This difference was confirmed by a repeated measures ANOVA, which resulted in F \= 5.503, p \= 0.0242. The distribution of average PAC values before the stimulus was centered slightly towards negative values \(mean: -0.05) and exhibited large tails in both positive and negative directions \(standard deviation: 0.05). In contrast, the distribution of PAC values after the stimulus was centered on positive values \(mean: 0.02) with narrower tails \(standard deviation: 0.01) \(see Figure Xa).

In addition, we found that this difference between PAC values were more pronounced in layer five and six \(see Figure Xb). As illustrated in Figure X, these two V1 layers were the only layers with F values higher than F-critical \(approximately 4). However , similar to Time frequency analysis, no further statistical tests

were applied to quantitatively evaluate the layer-specificity due to the variability in the number of channels across layers.

#figure([
#box(width: 656.1872909698997pt, image("images/pac_beforeVSafter.png"))
], caption: figure.caption(
position: bottom, 
[
Phase Amplitude Coupling \(PAC) values before and after stimulus. a) Histogram distribution of averaged normalized PAC values for frequency of phase \(2-7 HZ) and frequency for amplitude of \(25-80 HZ). The distribution is across all V1 channels \(N \= 2,262) and for two time period: after stimulus \(0-1 s) with color yellow and before stimulus \(-0.5, 0 s) with color light blue. The histograms are computed with 20 bins, and a kernel density estimate \(KDE) is overlaid with solid line. B) Boxplot illustration of before \(yellow) and after \(light blue) stimulus PAC values for each layer of V1, with number of channels per layer : layer1 \= 212, layer 2/3: 338, layer 4 \= 650, layer 5 \= 650, layer 6 \= 606. As you can see the difference is significant mainly in layer 5 and 6.
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


= References
<references>
#block[
] <refs>



#bibliography("references.bib")

