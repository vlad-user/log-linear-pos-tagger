\documentclass[]{article}
\usepackage{lmodern}
\usepackage{amssymb,amsmath}
\usepackage{ifxetex,ifluatex}
\usepackage{fixltx2e} % provides \textsubscript
\ifnum 0\ifxetex 1\fi\ifluatex 1\fi=0 % if pdftex
  \usepackage[T1]{fontenc}
  \usepackage[utf8]{inputenc}
\else % if luatex or xelatex
  \ifxetex
    \usepackage{mathspec}
  \else
    \usepackage{fontspec}
  \fi
  \defaultfontfeatures{Ligatures=TeX,Scale=MatchLowercase}
\fi
% use upquote if available, for straight quotes in verbatim environments
\IfFileExists{upquote.sty}{\usepackage{upquote}}{}
% use microtype if available
\IfFileExists{microtype.sty}{%
\usepackage[]{microtype}
\UseMicrotypeSet[protrusion]{basicmath} % disable protrusion for tt fonts
}{}
\PassOptionsToPackage{hyphens}{url} % url is loaded by hyperref
\usepackage[unicode=true]{hyperref}
\hypersetup{
            pdfborder={0 0 0},
            breaklinks=true}
\urlstyle{same}  % don't use monospace font for urls
\IfFileExists{parskip.sty}{%
\usepackage{parskip}
}{% else
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt plus 2pt minus 1pt}
}
\setlength{\emergencystretch}{3em}  % prevent overfull lines
\providecommand{\tightlist}{%
  \setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\setcounter{secnumdepth}{0}
% Redefines (sub)paragraphs to behave more like sections
\ifx\paragraph\undefined\else
\let\oldparagraph\paragraph
\renewcommand{\paragraph}[1]{\oldparagraph{#1}\mbox{}}
\fi
\ifx\subparagraph\undefined\else
\let\oldsubparagraph\subparagraph
\renewcommand{\subparagraph}[1]{\oldsubparagraph{#1}\mbox{}}
\fi

% set default figure placement to htbp
\makeatletter
\def\fps@figure{htbp}
\makeatother


\date{}

\begin{document}

\documentclass[]{article}
\usepackage{lmodern}
\usepackage{amssymb,amsmath}
\usepackage{ifxetex,ifluatex}
\usepackage{fixltx2e} % provides \textsubscript
\ifnum 0\ifxetex 1\fi\ifluatex 1\fi=0 % if pdftex
  \usepackage[T1]{fontenc}
  \usepackage[utf8]{inputenc}
\else % if luatex or xelatex
  \ifxetex
    \usepackage{mathspec}
  \else
    \usepackage{fontspec}
  \fi
  \defaultfontfeatures{Ligatures=TeX,Scale=MatchLowercase}
\fi
% use upquote if available, for straight quotes in verbatim environments
\IfFileExists{upquote.sty}{\usepackage{upquote}}{}
% use microtype if available
\IfFileExists{microtype.sty}{%
\usepackage[]{microtype}
\UseMicrotypeSet[protrusion]{basicmath} % disable protrusion for tt fonts
}{}
\PassOptionsToPackage{hyphens}{url} % url is loaded by hyperref
\usepackage[unicode=true]{hyperref}
\hypersetup{
            pdfborder={0 0 0},
            breaklinks=true}
\urlstyle{same}  % don't use monospace font for urls
\IfFileExists{parskip.sty}{%
\usepackage{parskip}
}{% else
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt plus 2pt minus 1pt}
}
\setlength{\emergencystretch}{3em}  % prevent overfull lines
\providecommand{\tightlist}{%
  \setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\setcounter{secnumdepth}{0}
% Redefines (sub)paragraphs to behave more like sections
\ifx\paragraph\undefined\else
\let\oldparagraph\paragraph
\renewcommand{\paragraph}[1]{\oldparagraph{#1}\mbox{}}
\fi
\ifx\subparagraph\undefined\else
\let\oldsubparagraph\subparagraph
\renewcommand{\subparagraph}[1]{\oldsubparagraph{#1}\mbox{}}
\fi

% set default figure placement to htbp
\makeatletter
\def\fps@figure{htbp}
\makeatother


\date{}

\begin{document}

\section{Log Linear Part-Of-Speech
tagger.}\label{log-linear-part-of-speech-tagger.}

\href{http://nbviewer.jupyter.org/github/vlad-user/log-linear-pos-tagger/blob/master/readme.pdf}{readme}

\end{document}
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{amssymb}
\usepackage{amsmath}
\usepackage{url}
\usepackage{graphicx} % table
\usepackage{graphicx} % figure
\usepackage{float} % figure
\usepackage{colortbl} % color of tables

\title{Training part-of-speech tagger using Log-Linear model}
%\author{vlad-user}
\date{November 2018}

\begin{document}

\maketitle
%\tableofcontents
\small
\section{Problem formulation}
Given input in from of $\big\{\{(w_j, t_j)\}_{j=1}^{n_i}\big\}_{i=1}^N$ where $w_j, t_j$ is $j$th word-tag pair in sentence $i$, we want to find $\theta\in\mathbb{R}^d$ that models empirically the marginal probability distribution $p(t|h)$ where $h$ is a history (a window within a sentence). 
The marginal probability distribution is defined as follows
\begin{align}
    p(t|h;\theta)=\frac{exp(\theta^Tf(h,t))}{\sum\limits_{t'\in S}exp(\theta^Tf(h,t'))}
\end{align}
where $f(h,t)\in\{0,1\}^d$ is a vector of indicator functions indicating whether a feature presents within a given history or not and $S$ is a set of all possible tags. 
By taking a logarithm we define a log linear function
\begin{align}
    log\big(p(t|h;\theta)\big)=\theta^Tf(h, t)-log(\sum\limits_{t'\in S}exp(\theta^Tf(h,t')))
\end{align}
By multiplying by $-1$ and introducing regularization parameter $\lambda > 0$ we have the following loss function to minimize
\begin{align}\label{loss_eqn}
    \mathcal{L}&=log\big(\sum\limits_{t'\in S}exp(\theta^Tf(h,t'))\big) - \theta^Tf(h, t) + \frac{\lambda}{2}\theta^T\theta
\end{align}
with gradients having following form
\begin{align}\label{grad_eqn}
    \frac{\partial\mathcal{L}}{\partial\theta_k}=\sum\limits_{i=1}^N\sum\limits_{t'\in S}f_k(h_i, t')p(t'|h_i;\theta)- \sum\limits_{i=1}^Nf_k(x_i, t_i)+\lambda\theta_k    
\end{align}
\section{Feature Extraction}
\subsection{Model 1}
As a template for  features for the first model we used more-or-less the same features as defined in \cite{ratnaparkhi1996maximum} except that we haven't considered rare/not rare words as separate cases. Rather we have filtered out all features that appeared less than two times\footnote{This is justified since we assuming that the dataset consist of i.i.d samples and it is large, meaning that occurences of features converge to the real distribution.}. Additionally, to reduce the number of features we haven't considered prefixes and suffixes of size 4.


\begin{table}[H]
    \centering
    \scalebox{0.8}{
    \begin{tabular}{ | l |l |p{4cm} |}
        \hline
        $w_{i+2}$ & A word that follows word $w_{i+1}$ \\ \hline
        $w_{i+1}$ & Next word \\ \hline
        $w_i$ & Current word at position i\\ \hline
        $w_{i-1}$ & Previous word\\ \hline
        $w_{i-2}$ & The word before word $w_{i-1}$ \\ \hline
        $t_{i-1}$ & Previous tag \\ \hline
        $t_{i-2}, t_{i-1}$ & Two previous tags \\ \hline
        prefix3 & Prefix of $w_i$ of size 3 letters\\ \hline
        prefix2 & Prefix of $w_i$ of size 2 letters\\ \hline
        prefix1 & Prefix of $w_i$ of size 1 letters\\ \hline
        suffix3 & Suffix of $w_i$ of size 3 letters\\ \hline
        suffix2 & Suffix of $w_i$ of size 2 letters\\ \hline
        suffix1 & Suffix of $w_i$ of size 1 letters\\ \hline
        has\_hyphen & 1 if $w_i$ contains hyphen, 0 otherwise \\ \hline
        has\_upper & 1 if $w_i$ contains uppercase letter, 0 otherwise \\ \hline
        has\_number & 1 if $w_i$ contains number, 0 otherwise \\ \hline
        
    \end{tabular}}
    \caption{Features for model 1.}
    \label{tab:features1}
\end{table}

Such vectorization resulted in 47401 features.


\subsection{Model 2}
Since train data for model 2 is much smaller, we haven't filtered out rare features and in addition to the first model we have added prefixes and suffixes of size 4. The table \ref{tab:features2} shows the whole list of features for model 2. Applying the table resulted in 30111 features.
\begin{table}[H]
    \centering
    \scalebox{0.8}{
    \begin{tabular}{ | l |l |p{4cm} |}
        \hline
        $w_{i+2}$ & A word that follows word $w_{i+1}$ \\ \hline
        $w_{i+1}$ & Next word \\ \hline
        $w_i$ & Current word at position i\\ \hline
        $w_{i-1}$ & Previous word\\ \hline
        $w_{i-2}$ & The word before word $w_{i-1}$ \\ \hline
        $t_{i-1}$ & Previous tag \\ \hline
        $t_{i-2}, t_{i-1}$ & Two previous tags \\ \hline
        prefix4 & Prefix of $w_i$ of size 4 \\ \hline
        prefix3 & Prefix of $w_i$ of size 3 letters\\ \hline
        prefix2 & Prefix of $w_i$ of size 2 letters\\ \hline
        prefix1 & Prefix of $w_i$ of size 1 letters\\ \hline
        suffix4 & Suffix of $w_i$ of size 4 \\ \hline
        suffix3 & Suffix of $w_i$ of size 3 letters\\ \hline
        suffix2 & Suffix of $w_i$ of size 2 letters\\ \hline
        suffix1 & Suffix of $w_i$ of size 1 letters\\ \hline
        has\_hyphen & 1 if $w_i$ contains hyphen, 0 otherwise \\ \hline
        has\_upper & 1 if $w_i$ contains uppercase letter, 0 otherwise \\ \hline
        has\_number & 1 if $w_i$ contains number, 0 otherwise \\ \hline
    \end{tabular}}
    \caption{Features for model 2.}
    \label{tab:features2}
\end{table}

\section{Training}
We used Limited-memory BFGS\footnote{\url{https://en.wikipedia.org/wiki/Limited-memory_BFGS}} algorithm defined in scipy-package\footnote{\url{https://docs.scipy.org/doc/scipy-1.1.0/reference/optimize.minimize-lbfgsb.html}}. The first model took around 20 hours to train and 40 minutes to train the second one.

\begin{figure}[H]
  \centering
  \includegraphics[width=\textwidth,height=\textheight,keepaspectratio]{train_loss_model1.png}
  \caption{Training loss (\ref{loss_eqn}) using BFGS optimization algorithm with $\lambda=0.005$ for model 1.}\label{bfgs_train_loss}
\end{figure}

\begin{figure}[H]
  \centering
  \includegraphics[width=\textwidth,height=\textheight,keepaspectratio]{train_loss_model2.png}
  \caption{Training loss (\ref{loss_eqn}) using BFGS optimization algorithm with $\lambda=0.005$ for model 2.}\label{sgd_train_loss}
\end{figure}

\section{Inference}
Given tokenized sentences as input $\big\{\{w_j\}_{j=1}^{n_i}\big\}_{i=1}^M$, for each sentence $i$, we want to find a sequence of tags  $\{t_j\}_{j=1}^{n_i}$, by assuming the probability of form
\begin{align}
    p(\{t_j\}_{j=1}^{n_i}|\{w_j\}_{j=1}^{n_i})=\prod\limits_{k=1}^{n_i}q(t_k|h_k), 
\end{align}
Using Viterbi algorithm, for each $(u, v)\in S_{m-1}\times S_m$ we define a reccurence relation
\begin{align}
    \pi(m, u, v)=max_{t\in S_{m-2}}\Big\{\pi(m-1,t, u)q(v|h_m(t, u, \{w_j\}_{j=1}^{n_i}, m))\Big\}
\end{align}
where $S_m$ is a set of all possible tags at the position $m$ within a sentence.

It is easy to see that for each element $\pi(m, u, v)$ we calculate probability $q(v|h_m(t, u, \{w_j\}_{j=1}^{n_i}, m)$ which results in $|S_{m-2}\times S_{m-1}\times S_m|$ computations for each word and overall complexity for a sentence with $n$ words results in $\mathcal{O}(|S|^3n)$ calculations. Although it does significantly improves the brute-force  $\mathcal{O}(|S|^n)$ complexity, having a large set of tags (in our case 44) is still computationally expensive. 
\section{Results}
\subsection{Model 1}
\begin{figure}[H]
  \centering
  \includegraphics[width=\textwidth,height=\textheight,keepaspectratio]{cm.png}
  \caption{Confusion matrix for model 1.}\label{cm}
\end{figure}
Figure (\ref{cm}) shows the confusion matrix for model 1. The tags that have the worst performance are: IN, NN, NNP, DT, NNS, CD, `,`, `.`, RB, VBD. The `,` and `.` can be improved by adding additional feature indicating whether a word contains punctuation symbols. For other tags, there are multiple ways to improve their predictions. One way would be to understand what causes such performance: Analyze histories for such cases and look for features that cause the model to fail. For instance, it may be due to fact that  some prefixes do improve the performance and some prefixes worsens it so one should create a list of blacklisted prefixes that won't be included as features (or maybe it is better to remove prefixes of some length completely). Another way, similar to \cite{collins2002ranking}, is to use some form of ranking to improve the model: Take the output candidates from the first model and use them as input to the second model in an effort to improve the performance.
The accuracy for test dataset that we have achieved is 0.943. Figure (\ref{accuracy}) shows its the screenshot. It also shows that the average time for inference for a single sentence is 57 seconds.
\begin{figure}[H]
  \centering
  \includegraphics[width=\textwidth,height=\textheight,keepaspectratio]{accuracy.png}
  \caption{Screenshot of the accuracy for model 1.}\label{accuracy}
\end{figure}

\subsection{Model 2}
Since the size of the data for model 2 is small, to predict its performance we've trained the model 2 ten times each time randomly splitting the dataset by taking out 20\% for testing and training with the rest 80\%. 
\begin{table}[H]
    \centering
    \scalebox{0.6}{
    \begin{tabular}{ |l |l |l |l |l |l |l |l |l |l |l |l |l | p{4cm} |}
        \hline
        simulation & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 & mean & err\\ \hline
        accuracy & 0.979 & 0.933 & 0.929 & 0.945 & 0.936 & 0.919 & 0.939 & 0.951 & 0.969 & 0.921 & 0.942 & 0.019\\ 
    \end{tabular}}
    \caption{Validation simulations for model 2.}
    \label{tab:cross_valid_model2}
\end{table}

Table \ref{tab:cross_valid_model2} shows results for 10 simulations, their mean and standard deviation. The generalization error should be somewhere around 0.942. Taking the worst case, we project the accuracy to be $0.942 -0.019=0.923$.

\bibliographystyle{plain}
\bibliography{bibliography.bib}
\end{document}

\end{document}
