


const METRICS = [
    "Task Fulfilment / Relevance", "Usefulness & Actionability", "Factual Accuracy & Verifiability",
    "Efficiency / Depth & Completeness", "Reasoning Quality / Transparency", "Tone & Likeability",
    "Adaptation to Context", "Safety & Bias Avoidance", "Structure & Formatting & UX Extras", "Creativity"
];

const DATASET_FILES = {
    alpaca: "alpaca_500_sample100.json",
    gsm8k: "gsm8k_500_sample100.json",
    mmlu: "mmlu_500_sample100.json"
};

const DATA_PATHS = {
    instructions: (dataset) => `/samples/prompts_paraphrases/${DATASET_FILES[dataset]}`,
    scores: (dataset, model) => `/samples/metric_scores/${dataset}/${model}_sample100.json`,
    examples: () => "/site_data/high_performing_examples.json",
    styleTooltips: () => "/site_data/style_tooltip_examples.json",
    aggregates: () => "/site_data/full_aggregate_metrics.json"
};

let state = {
    currentDataset: 'alpaca',
    currentModel: 'gemma-2-2b-it',
    instructions: [],
    scores: [],
    promptIndex: new Map(),
    examples: [],
    exampleBundle: null,
    aggregateBundle: null,
    styleTooltipExamples: {},
    aggregatedData: {}, // Holds processed averages, stddevs, etc.
    sampleAggregatedData: {},
    corpusSignals: {},
    topStyleExampleKeys: new Set(),
    spiderChart: null,
    barChart: null,
    isLoading: true,
};



const PARAPHRASE_FAMILIES = {
  
  original: ["instruction_original"],

  
  english: [
    "instruct_american_english",
    "instruct_australian_english",
    "instruct_british_english"
  ],
  language: [
    "instruct_chinese_simplified","instruct_french","instruct_spanish",
    "instruct_german","instruct_esperanto","instruct_klingon","instruct_scots",
    "instruct_hinglish","instruct_singlish","instruct_spanglish"
  ],

  
  informal: [
    "instruct_aave","instruct_casual","instruct_casual_chat","instruct_colloquial",
    "instruct_leet_speak","instruct_cockney","instruct_gamer_slang",
    "instruct_gaming_jargon","instruct_slang_heavy","instruct_informal"
  ],

  
  formal: [
    "instruct_authoritative","instruct_bureaucratic","instruct_formal_academic",
    "instruct_formal_business","instruct_formal_memo","instruct_very_formal",
    "instruct_legalease","instruct_legalese","instruct_legal_jargon",
    "instruct_modal_may","instruct_modal_must","instruct_modal_should"
  ],

  
  positive_tone: [
    "instruct_apologetic","instruct_apology","instruct_polite_request",
    "instruct_friendly","instruct_warm","instruct_enthusiastic","instruct_hopeful",
    "instruct_positive","instruct_lighthearted","instruct_confident",
    "instruct_helpful_meaning_reinforing_characters"
  ],

  
  negative_tone: [
    "instruct_vulgar","instruct_profane","instruct_rude","instruct_cynical",
    "instruct_sarcastic","instruct_sardonic","instruct_ironic","instruct_deadpan",
    "instruct_insulting","instruct_skeptical","instruct_melancholy"
  ],

  
  humor: [
    "instruct_humorous","instruct_joke","instruct_pun","instruct_pun_based",
    "instruct_witty","instruct_silly","instruct_playful","instruct_rap_verse",
    "instruct_absurdist","instruct_surreal"
  ],
  poetic: ["instruct_haiku","instruct_poetic","instruct_lyrical","instruct_shakespeare"],

  
  professional_jargon: [
    "instruct_finance_jargon","instruct_medical_jargon","instruct_sports_jargon",
    "instruct_fashion_jargon","instruct_culinary_jargon","instruct_physics_jargon",
    "instruct_software_jargon","instruct_technical","instruct_jargon",
    "instruct_marketing","instruct_marketing_speak"
  ],

  
  typo: [
    "instruct_misplaced_commas","instruct_missing_bracket",
    "instruct_missing_bracket_and_quote","instruct_missing_quote",
    "instruct_one_typo_punctuation","instruct_two_typos_punctuation",
    "instruct_three_typos_punctuation","instruct_typo_adjacent",
    "instruct_typo_extra_letter","instruct_typo_extra_space","instruct_typo_homophone",
    "instruct_typo_missing_letter","instruct_typo_missing_space",
    "instruct_typo_missing_vowels","instruct_typo_random","instruct_typo_repeated_letters",
    "instruct_typo_swap","instruct_typo_swap_and_punctuation",
    "instruct_typo_swap_and_transpose_and_punctuation","instruct_typo_transpose",
    "instruct_typo_wrong_letter","instruct_edit_typo","instruct_key_smash",
    
    "instruct_all_caps_and_typo","instruct_all_caps_and_typo_and_missing_bracket",
    "instruct_all_caps_and_typo_and_missing_bracket_and_random_characters",
    "instruct_curly_quotations_and_typo","instruct_curly_quotations_and_missing_bracket_and_typo",
    "instruct_curly_quotations_and_missing_bracket_and_typo_and_random_characters",
    "instruct_emoji_and_typo","instruct_emoji_and_typo_and_missing_bracket",
    "instruct_emoji_and_typo_and_random_question_marks","instruct_emoticon_and_typo",
    "instruct_emoticon_and_typo_and_missing_bracket","instruct_emoticon_and_typo_and_random_exclamations",
    "instruct_random_linebreaks_and_typo_and_missing_bracket",
    "instruct_random_linebreaks_and_typo_and_missing_bracket_and_many_exclamations",
    "instruct_random_linebreaks_and_typo_and_missing_bracket_and_wrong_punctuation",
    "instruct_random_linebreaks_and_typo_and_missing_bracket_and_wrong_punctuation_and_extra_space",
    "instruct_random_linebreaks_and_typo_and_missing_random_characters",
    "instruct_random_linebreaks"
  ],
  caps_case: [
    "instruct_all_caps","instruct_no_caps","instruct_random_caps","instruct_no_contractions"
  ],

  
  emoji: [
    "instruct_emoji","instruct_emoji_only","instruct_emoji_and_typo",
    "instruct_emoji_and_typo_and_missing_bracket","instruct_emoji_and_typo_and_random_question_marks",
    "instruct_emoticon","instruct_emoticon_and_typo",
    "instruct_emoticon_and_typo_and_missing_bracket",
    "instruct_emoticon_and_typo_and_random_exclamations"
  ],

  
  encoding: [
    "instruct_base64","instruct_morse_code","instruct_rot13","instruct_reversed_text",
    "instruct_small_hex_blob","instruct_scientific_notation",
    "instruct_musical_notation","instruct_roman_numeral"
  ],

  
  markup: [
    "instruct_markdown_bold","instruct_markdown_italic","instruct_markdown_bold_and_italic",
    "instruct_markdown_quote","instruct_markdown_doc","instruct_helpful_markdown_structure",
    "instruct_html_tags","instruct_several_html_tags","instruct_html_comment",
    "instruct_output_html","instruct_output_markdown","instruct_yaml_block","instruct_code_fence"
  ],
  data_format: [
    "instruct_csv_line","instruct_csv_row","instruct_output_csv","instruct_json_format",
    "instruct_output_json","instruct_output_yaml","instruct_output_sql",
    "instruct_output_python","instruct_react_tool_calls"
  ],
  list_format: [
    "instruct_90char_bullet","instruct_bullet_list","instruct_bulleted_outline",
    "instruct_numbered_list","instruct_numbered_steps","instruct_checklist",
    "instruct_checklist_markdown","instruct_table_layout","instruct_comparison_table"
  ],

  
  condense_expand: [
    "instruct_condensed_then_expand",
    "instruct_condensed_then_expand_with_examples",
    "instruct_condensed_then_expand_with_examples_and_explanations",
    "instruct_condensed_then_expand_with_examples_and_explanations_and_summary",
    "instruct_condensed_then_expand_with_examples_and_explanations_and_summary_and_risks",
    "instruct_condensed_then_expand_with_examples_and_explanations_and_summary_and_risks_and_benefits",
    "instruct_condensed_then_expand_with_examples_and_explanations_and_summary_and_risks_and_benefits_and_references",
    "instruct_condensed_then_expand_with_examples_and_explanations_and_summary_and_risks_and_benefits_and_references_and_citations",
    "instruct_condensed_then_expand_with_examples_and_explanations_and_summary_and_risks_and_benefits_and_references_and_citations_and_counterarguments",
    "instruct_condensed_then_expand_with_examples_and_explanations_and_summary_and_risks_and_benefits_and_references_and_citations_and_counterarguments_and_rebuttals",
    "instruct_condensed_then_expand_with_examples_and_explanations_and_summary_and_risks_and_benefits_and_references_and_citations_and_counterarguments_and_rebuttals_and_analogies",
    "instruct_condensed_then_expand_with_examples_and_explanations_and_summary_and_risks_and_benefits_and_references_and_citations_and_counterarguments_and_rebuttals_and_analogies_and_metaphors",
    "instruct_summary_then_detail","instruct_risks_and_benefits",
    "instruct_with_summary","instruct_with_tldr_summary"
  ],

  
  reasoning: [
    "instruct_plan_execute_reflect","instruct_role_expert_cot","instruct_role_expert_cot_with_examples",
    "instruct_role_expert_cot_with_examples_and_explanations",
    "instruct_role_expert_cot_with_examples_and_explanations_and_summary",
    "instruct_role_expert_cot_with_examples_and_explanations_and_summary_and_risks",
    "instruct_step_rationale","instruct_self_consistency","instruct_with_step_by_step",
    "instruct_dynamic_quiz","instruct_with_examples","instruct_with_examples_and_explanations",
    "instruct_with_helpful_explanations","instruct_with_detailed_instructions",
    "instruct_with_technical_details","instruct_fact_check_inline","instruct_evidence_cited_md",
    "instruct_exact_numbers"
  ],

  
  question_form: [
    "instruct_direct_question","instruct_indirect_question","instruct_choice_question",
    "instruct_nested_question","instruct_wh_question","instruct_tag_question",
    "instruct_rhetorical_question","instruct_double_negative"
  ],

  
  medium: [
    "instruct_email","instruct_sms","instruct_sms_abbrev","instruct_tweet",
    "instruct_news_headline","instruct_emergency_alert","instruct_urgent",
    "instruct_forum_quote","instruct_journalist_interview","instruct_tech_support_ticket",
    "instruct_timestamped_chat","instruct_qa_script","instruct_inline_ad",
    "instruct_inline_url","instruct_hashtags"
  ],

  
  punctuation: [
    "instruct_exclamation","instruct_interrobang","instruct_ellipsis_style",
    "instruct_em_dash_break","instruct_extra_punct","instruct_no_punct"
  ],

  
  length_control: [
    "instruct_few_words","instruct_fewest_words","instruct_single_sentence",
    "instruct_two_sentence","instruct_short_paragraph","instruct_multi_paragraph"
  ],

  
  creative: [
    "instruct_surreal","instruct_absurdist","instruct_garden_path","instruct_paradox",
    "instruct_paradox_statement","instruct_recursive_self_reference",
    "instruct_ambiguous_scope","instruct_hypothetical_if","instruct_malapropism",
    "instruct_litotes","instruct_parenthetical_aside","instruct_sentence_fragment",
    "instruct_nested_parentheticals","instruct_pseudo_cleft","instruct_topicalization",
    "instruct_passive_voice","instruct_nominalization","instruct_inversion",
    "instruct_see_attached_diagram"
  ],

  
  rephrase: [
    "instruct_acronyms_spelled_out","instruct_advertisement","instruct_chemical_smiles",
    "instruct_child_directed","instruct_cleft_it_is","instruct_command",
    "instruct_contractions","instruct_contradictory_ask","instruct_coord_to_subord",
    "instruct_exam_prompt","instruct_expert_consensus","instruct_fuzzy_numbers",
    "instruct_future_tense","instruct_greeting","instruct_impersonal_one_should",
    "instruct_indirect_relay","instruct_meta_question","instruct_minimalist",
    "instruct_might_be_wrong","instruct_negated","instruct_news_headline",
    "instruct_no_spaces","instruct_oxford_comma","instruct_redundant_waffle",
    "instruct_regex_pattern","instruct_rubric_scored","instruct_salesy",
    "instruct_second_person","instruct_self_deprecating","instruct_sic_marker",
    "instruct_spoiler_bars","instruct_sql_snippet","instruct_statement",
    "instruct_study_setup","instruct_suggestion","instruct_therapy_session",
    "instruct_third_person","instruct_validator_pass","instruct_with_additional_context",
    "instruct_with_analogies","instruct_with_case_studies","instruct_with_citations",
    "instruct_with_counterarguments","instruct_with_emotional_appeal",
    "instruct_with_metaphors","instruct_with_personal_touch","instruct_with_rebuttals",
    "instruct_with_research_paper","instruct_with_similes","instruct_with_statistics",
    "instruct_with_stream_of_consciousness","instruct_yes_no"
  ]
};


function formatParaphraseStyle(styleKey) {
    if (styleKey === 'instruction_original') return 'Original';

    return styleKey
        .replace(/^instruct_/, '')         // remove prefix
        .split('_')                        // split by underscore
        .map(word =>
            /^[A-Z0-9]+$/.test(word) ? word : word.charAt(0).toUpperCase() + word.slice(1)
        )
        .join(' ');
}

function formatFamilyLabel(famKey) {
  return famKey
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase());
}



document.addEventListener('DOMContentLoaded', init);


function init() {
    console.log("CALIPER Explorer Initializing...");
    setupEventListeners();
    populateStaticElements();
    loadAndProcessData();
}


function setupEventListeners() {
    document.getElementById('dataset-select').addEventListener('change', handleDatasetChange);
    document.getElementById('model-select').addEventListener('change', handleModelChange);

    document.querySelectorAll('.nav-button').forEach(button => {
        button.addEventListener('click', handleNavClick);
    });

    document.getElementById('update-spider-chart').addEventListener('click', renderSpiderChart);
    document.getElementById('update-bar-chart').addEventListener('click', renderBarChart);
    
    document.getElementById('ranking-metric-select').addEventListener('change', renderRankingList);

    document.getElementById('search-button').addEventListener('click', handleSearch);

    document.getElementById('metric-select')
            .addEventListener('change', renderBarChart);
}


function populateStaticElements() {
    const rankingSelect = document.getElementById('ranking-metric-select');
    METRICS.forEach((metric, index) => {
        const option = document.createElement('option');
        option.value = index;
        option.textContent = metric;
        rankingSelect.appendChild(option);
    });

    const metricSelect = document.getElementById('metric-select');
    if (metricSelect) {
        METRICS.forEach((metric, idx) => {
            const opt = new Option(metric, idx);
            metricSelect.appendChild(opt);
        });
        metricSelect.selectedIndex = 0;          // default = first metric
    }
}




async function loadAndProcessData() {
    setLoading(true);
    try {
        const instructionsPath = DATA_PATHS.instructions(state.currentDataset);
        const scoresPath = DATA_PATHS.scores(state.currentDataset, state.currentModel);

        const [instructionsRes, scoresRes] = await Promise.all([
            fetch(instructionsPath),
            fetch(scoresPath)
        ]);

        if (!instructionsRes.ok) throw new Error(`Failed to load instructions: ${instructionsRes.statusText}`);
        if (!scoresRes.ok) throw new Error(`Failed to load scores: ${scoresRes.statusText}`);

        state.instructions = await instructionsRes.json();
        state.scores = await scoresRes.json();
        if (!state.exampleBundle || !state.aggregateBundle) {
            const [examplesRes, tooltipRes, aggregatesRes] = await Promise.all([
                fetch(DATA_PATHS.examples()),
                fetch(DATA_PATHS.styleTooltips()),
                fetch(DATA_PATHS.aggregates())
            ]);
            if (!examplesRes.ok) throw new Error(`Failed to load examples: ${examplesRes.statusText}`);
            if (!tooltipRes.ok) throw new Error(`Failed to load style tooltips: ${tooltipRes.statusText}`);
            if (!aggregatesRes.ok) throw new Error(`Failed to load full aggregate metrics: ${aggregatesRes.statusText}`);
            state.exampleBundle = await examplesRes.json();
            state.styleTooltipExamples = (await tooltipRes.json()).examples || {};
            state.aggregateBundle = await aggregatesRes.json();
        }
        state.examples = state.exampleBundle?.[state.currentDataset]?.[state.currentModel] || [];

        processData();
        renderAll();

    } catch (error) {
        console.error("Error loading data:", error);
        document.getElementById('overview-table-container').innerHTML = `<p style="color:red;">Error: ${error.message}. Please check the published sample data paths.</p>`;
    } finally {
        setLoading(false);
    }
}


function processData() {
    state.promptIndex = new Map();
    state.instructions.forEach(item => {
        state.promptIndex.set(getPromptKey(item), item);
    });

    const sampleAggregated = buildSampleAggregates(state.scores);
    const fullAggregated = state.aggregateBundle?.[state.currentDataset]?.[state.currentModel] || null;

    state.sampleAggregatedData = sampleAggregated;
    state.aggregatedData = mergeFullAggregatesWithSampleScores(fullAggregated, sampleAggregated);
    state.corpusSignals = buildCorpusSignals();
    console.log("Processed full aggregate data:", state.aggregatedData);
}

function buildSampleAggregates(scoreRows) {
    const aggregated = {};
    const paraphraseKeys = new Set();

    scoreRows.forEach(item => {
        Object.keys(item).forEach(key => {
            if (key !== 'prompt_count' && key !== 'prompt_id') {
                paraphraseKeys.add(key);
            }
        });
    });

    paraphraseKeys.forEach(key => {
        aggregated[key] = {
            scores: Array(METRICS.length).fill(0).map(() => []), // Store all scores for std dev
            averages: Array(METRICS.length).fill(0),
            stdDevs: Array(METRICS.length).fill(0),
            overallAverage: 0,
            count: 0,
        };
    });

    scoreRows.forEach(item => {
        paraphraseKeys.forEach(key => {
            if (item[key] && Array.isArray(item[key])) {
                aggregated[key].count++;
                item[key].forEach((score, index) => {
                    aggregated[key].scores[index].push(score);
                });
            }
        });
    });

    paraphraseKeys.forEach(key => {
        if (aggregated[key].count > 0) {
            for (let i = 0; i < METRICS.length; i++) {
                const scoresList = aggregated[key].scores[i];
                if(scoresList.length > 0) {
                    const avg = calculateAverage(scoresList);
                    aggregated[key].averages[i] = avg;
                    aggregated[key].stdDevs[i] = calculateStdDev(scoresList, avg);
                }
            }
            aggregated[key].overallAverage = calculateAverage(aggregated[key].averages);
        }
    });

    return aggregated;
}

function mergeFullAggregatesWithSampleScores(fullAggregated, sampleAggregated) {
    if (!fullAggregated) return sampleAggregated;

    const merged = {};
    Object.entries(fullAggregated).forEach(([style, data]) => {
        const sampleData = sampleAggregated[style];
        merged[style] = {
            count: data.count || 0,
            averages: data.averages || Array(METRICS.length).fill(0),
            stdDevs: data.stdDevs || Array(METRICS.length).fill(0),
            overallAverage: data.overallAverage ?? calculateAverage(data.averages || []),
            scores: sampleData?.scores || Array(METRICS.length).fill(0).map(() => []),
            sampleCount: sampleData?.count || 0,
        };
    });

    Object.entries(sampleAggregated).forEach(([style, data]) => {
        if (!merged[style]) merged[style] = data;
    });

    return merged;
}


function renderAll() {
    renderOverviewPage();
    renderRankingPage();
    renderSearchPage();
    
    const activeNav = document.querySelector('.nav-button.active');
    switchPage(activeNav.id);
}

function renderOverviewPage() {
    renderOverviewTable();
    populateChartSelectors();
    renderSpiderChart();
    renderBarChart();
}

function renderRankingPage() {
    renderRankingList();
}

function renderSearchPage() {
    renderTopPerformingStyles();
    document.getElementById('example-suggestions-container').innerHTML = '';
}



function renderOverviewTable() {
    const container = document.getElementById('overview-table-container');
    let html = '<table id="overview-table"><thead><tr><th>Paraphrase&nbsp;Style</th>';
    METRICS.forEach(m => html += `<th>${m}</th>`);
    html += '</tr></thead><tbody>';

    for (const [family, styleList] of Object.entries(PARAPHRASE_FAMILIES)) {
        const presentStyles = styleList.filter(s => state.aggregatedData[s]);
        if (presentStyles.length === 0) continue;

        const familyMeans = METRICS.map((_, idx) =>
            calculateAverage(
                presentStyles.map(s => state.aggregatedData[s].averages[idx])
            )
        );

        html += `<tr class="summary-row" data-family="${family}">
            <td><strong>${formatFamilyLabel(family)}</strong></td>`;
        familyMeans.forEach(avg =>
            html += `<td style="background:${scoreToColor(avg)}">${avg.toFixed(2)}</td>`
        );
        html += '</tr>';

        presentStyles.forEach(style => {
                html += `<tr class="detail-row" data-family="${family}">
                            <td style="padding-left:2rem;">${formatParaphraseStyle(style)}</td>`;
            state.aggregatedData[style].averages.forEach(a =>
                html += `<td style="background:${scoreToColor(a)}">${a.toFixed(2)}</td>`
            );
            html += '</tr>';
        });
    }

    html += '</tbody></table>';
    container.innerHTML = html;

    document.querySelectorAll('.summary-row').forEach(row => {
        row.addEventListener('click', () => {
            const fam = row.dataset.family;
            document
              .querySelectorAll(`.detail-row[data-family="${fam}"]`)
              .forEach(d => d.style.display = d.style.display === 'table-row' ? 'none' : 'table-row');
        });
    });
}


function renderRankingList() {
    const container = document.getElementById('ranking-container');
    const metricIndex = Number(document.getElementById('ranking-metric-select').value);

    if (!state.aggregatedData || Object.keys(state.aggregatedData).length === 0) return;

    const originalData = state.aggregatedData.instruction_original;
    const originalScore = originalData?.averages?.[metricIndex] ?? null;
    const sortedStyles = Object.entries(state.aggregatedData)
        .filter(([, data]) => data.count > 0)
        .sort(([, a], [, b]) => b.averages[metricIndex] - a.averages[metricIndex]);

    let listHtml = '';
    sortedStyles.forEach(([key, data]) => {
        listHtml += renderRankingItem(key, data, metricIndex, originalScore, key === 'instruction_original');
    });
    container.innerHTML = listHtml;
}

function renderRankingItem(key, data, metricIndex, originalScore, isBaseline) {
    const score = data.averages[metricIndex];
    const barWidth = Math.max(2, Math.min(100, (score / 10) * 100));
    const tooltip = getStyleTooltip(key);
    const delta = !isBaseline && Number.isFinite(originalScore) ? score - originalScore : null;
    const deltaClass = delta === null ? 'baseline' : delta >= 0 ? 'positive' : 'negative';
    const deltaText = delta === null ? '' : `${delta >= 0 ? '+' : ''}${delta.toFixed(2)} vs original`;

    return `
        <div class="ranking-item${isBaseline ? ' baseline-ranking-item' : ''}">
            <div class="ranking-label tooltip-label" data-tooltip="${escapeAttribute(tooltip)}">${formatParaphraseStyle(key)}</div>
            <div class="ranking-bar-container">
                <div class="ranking-bar" style="width: ${barWidth}%;">${score.toFixed(2)}</div>
            </div>
            <div class="ranking-score-meta ${deltaClass}">${deltaText}</div>
        </div>
    `;
}


function renderTopPerformingStyles() {
    const container = document.getElementById('top-styles-container');
    state.topStyleExampleKeys = new Set();
    if (!state.aggregatedData || Object.keys(state.aggregatedData).length === 0) {
        container.innerHTML = "<p>Style data is not available.</p>";
        return;
    }

    const styles = getRecommendableStyles().slice(0, 8);

    container.innerHTML = styles.map(([style, data], index) => {
        const example = findExampleForStyle(style, state.topStyleExampleKeys);
        state.topStyleExampleKeys.add(getExampleKey(example));
        return `
            <article class="style-card">
                <div class="style-rank">${index + 1}</div>
                <div class="style-card-body">
                    <h3>${formatParaphraseStyle(style)}</h3>
                    <div class="score-pills">
                        <span>TF ${data.averages[0].toFixed(2)}</span>
                        <span>Overall ${data.overallAverage.toFixed(2)}</span>
                        <span>N=${data.count}</span>
                    </div>
                    <p class="prompt-text">${escapeHTML(trimText(example.prompt, 320))}</p>
                </div>
            </article>
        `;
    }).join('');
}


function setLoading(isLoading) {
    state.isLoading = isLoading;
    document.getElementById('loader').style.display = isLoading ? 'block' : 'none';
    document.querySelector('main').style.display = isLoading ? 'none' : 'block';
}



function handleDatasetChange(e) {
    state.currentDataset = e.target.value;
    console.log(`Dataset changed to: ${state.currentDataset}`);
    loadAndProcessData();
}

function handleModelChange(e) {
    state.currentModel = e.target.value;
    console.log(`Model changed to: ${state.currentModel}`);
    loadAndProcessData();
}

function handleNavClick(e) {
    const targetId = e.currentTarget.id;
    switchPage(targetId);
}

function handleSearch() {
    const query = document.getElementById('search-input').value;
    analyzeAndDisplaySearch(query);
}


function switchPage(targetNavId) {
    document.querySelectorAll('.nav-button').forEach(btn => btn.classList.remove('active'));
    document.getElementById(targetNavId).classList.add('active');

    const pageId = targetNavId.replace('nav-', '') + '-page';
    document.querySelectorAll('.page').forEach(page => {
        page.style.display = page.id === pageId ? 'block' : 'none';
    });
}




function populateChartSelectors() {
    const spiderSelect = document.getElementById('spider-select');
    const barSelect = document.getElementById('bar-select');
    spiderSelect.innerHTML = '';
    barSelect.innerHTML = '';

    const sortedStyles = Object.keys(state.aggregatedData).sort();

    sortedStyles.forEach(key => {
        if (state.aggregatedData[key].count > 0) {
            const option = new Option(formatParaphraseStyle(key), key);
            spiderSelect.add(option.cloneNode(true));
            barSelect.add(option);
        }
    });

    const defaultSelections = ['instruction_original', 'instruct_apologetic', 'instruct_direct_question'].filter(s => sortedStyles.includes(s));
    for (const option of spiderSelect.options) {
        if (defaultSelections.includes(option.value)) option.selected = true;
    }
     for (const option of barSelect.options) {
        if (defaultSelections.includes(option.value)) option.selected = true;
    }
}


function renderSpiderChart() {
    const ctx = document.getElementById('spider-chart').getContext('2d');
    const selectedOptions = Array.from(document.getElementById('spider-select').selectedOptions).map(opt => opt.value);
    
    const datasets = selectedOptions.map((key, index) => {
        const color = `hsl(${(index * 100) % 360}, 70%, 50%)`;
        return {
            label: formatParaphraseStyle(key),
            data: state.aggregatedData[key].averages,
            borderColor: color,
            backgroundColor: `${color}33`, // semi-transparent fill
            pointBackgroundColor: color,
            pointBorderColor: '#fff',
            pointHoverBackgroundColor: '#fff',
            pointHoverBorderColor: color
        };
    });

    if (state.spiderChart) state.spiderChart.destroy();
    
    state.spiderChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: METRICS.map(m => m.split(' ')[0]), // Use shorter labels for chart
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                r: {
                    angleLines: { display: true },
                    suggestedMin: 0,
                    suggestedMax: 10,
                    pointLabels: {
                        font: { size: 10 }
                    }
                }
            },
            plugins: {
                legend: { position: 'top' }
            }
        }
    });
}


function renderBarChart() {          // ← keep the old name: no other code breaks
    const ctx = document.getElementById('bar-chart').getContext('2d');

    const metricIdx = Number(document.getElementById('metric-select').value);
    const styles     = Array.from(document.getElementById('bar-select').selectedOptions)
                            .map(o => o.value);

    const labels = [];
    const datasetsData = [];
    const bgColors = [];

    styles.forEach((style, i) => {
        const rawScores = state.aggregatedData?.[style]?.scores?.[metricIdx] || [];
        if (rawScores.length === 0) return;     // skip empty ones

        labels.push(formatParaphraseStyle(style));
        datasetsData.push(rawScores);           // plugin calculates quartiles itself
        bgColors.push(`hsl(${(i * 75) % 360}, 60%, 70%)`);
    });

    if (state.barChart) state.barChart.destroy();

    state.barChart = new Chart(ctx, {
        type: 'boxplot',
        data: {
            labels: labels,
            datasets: [{
                label: METRICS[metricIdx],
                data: datasetsData,
                backgroundColor: bgColors,
                borderColor: bgColors,
                borderWidth: 1,
                outlierColor: '#666',
                padding: 10,
                itemRadius: 0,
                showMean: true,
                meanColor: '#000'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,   // let CSS decide the height
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: {
                    label(ctx) {
                        const v = ctx.raw;
                        return [
                            `min: ${v.min}`,
                            `Q1 : ${v.q1}`,
                            `median: ${v.median}`,
                            `Q3 : ${v.q3}`,
                            `max: ${v.max}`,
                            `mean: ${v.mean.toFixed(2)}`
                        ];
                    }
                }}
            },
            scales: {
                y: { beginAtZero: true, max: 10 },
                            x : {
                ticks : {
                    font  : {
                        size  : 14,
                        weight: '500'         // 400=normal, 600=semibold, 700=bold
                    },
                    color : '#222',           // darker text
                    padding : 4,              // little breathing room above axis
                    maxRotation : 90,         // angle if names are long
                    minRotation : 45
                }
            }
            }
        }
    });
}




const STOPWORDS = new Set([
    "a","an","and","are","as","at","be","but","by","can","could","for","from","has","have",
    "how","i","in","into","is","it","its","me","my","of","on","or","our","please","should",
    "that","the","their","then","there","these","this","to","use","using","was","we","what",
    "when","where","which","who","why","will","with","would","you","your"
]);

const STYLE_FEATURES = [
    { style: "instruct_direct_question", label: "direct task request", risk: "low", patterns: [/^\s*(give|list|explain|write|generate|summarize|classify|compare|solve|calculate|answer)\b/i, /\bwhat\b.*\?$/i] },
    { style: "instruct_polite_request", label: "polite request", risk: "low", patterns: [/\b(please|could you|would you|kindly|can you)\b/i] },
    { style: "instruct_apologetic", label: "apologetic framing", risk: "low", patterns: [/\b(sorry|apologize|bother|trouble|if it'?s not too much)\b/i] },
    { style: "instruct_formal_academic", label: "formal or academic language", risk: "low", patterns: [/\b(furthermore|therefore|consequently|articulate|delineate|evaluate|analyze|synthesize)\b/i] },
    { style: "instruct_bureaucratic", label: "bureaucratic wording", risk: "medium", patterns: [/\b(pursuant|aforementioned|enumerate|comprehensive|classified as|provide a comprehensive)\b/i] },
    { style: "instruct_legalese", label: "legal wording", risk: "medium", patterns: [/\b(pursuant|liability|statute|hereby|jurisdiction|compliance|regulatory|whereas)\b/i] },
    { style: "instruct_finance_jargon", label: "finance jargon", risk: "medium", patterns: [/\b(asset|portfolio|risk-adjusted|capital|liquidity|return on|valuation|exposure)\b/i] },
    { style: "instruct_medical_jargon", label: "medical jargon", risk: "medium", patterns: [/\b(diagnosis|clinical|symptom|treatment|patient|pathology|therapeutic)\b/i] },
    { style: "instruct_software_jargon", label: "software jargon", risk: "low", patterns: [/\b(api|algorithm|runtime|schema|database|function|code|implementation|debug)\b/i] },
    { style: "instruct_casual", label: "casual phrasing", risk: "low", patterns: [/\b(hey|yo|gonna|wanna|kinda|sorta|stuff|thingy)\b/i] },
    { style: "instruct_slang_heavy", label: "slang-heavy phrasing", risk: "medium", patterns: [/\b(bro|fam|lit|vibe|lowkey|highkey|no cap)\b/i] },
    { style: "instruct_sarcastic", label: "sarcastic or cynical tone", risk: "medium", patterns: [/\b(sure|obviously|totally|as if|groundbreaking|supposedly)\b/i] },
    { style: "instruct_profane", label: "profane wording", risk: "medium", patterns: [/\b(hell|damn|shit|fuck|fucking)\b/i] },
    { style: "instruct_insulting", label: "insulting wording", risk: "high", patterns: [/\b(idiot|stupid|moron|dumb|bet you can'?t)\b/i] },
    { style: "instruct_humorous", label: "humor or joke setup", risk: "medium", patterns: [/\b(joke|funny|pun|riddle|humor|laugh)\b/i] },
    { style: "instruct_haiku", label: "poetic or verse constraint", risk: "high", patterns: [/\b(haiku|poem|poetic|verse|rhyme|sonnet)\b/i] },
    { style: "instruct_rap_verse", label: "rap or song-like prompt", risk: "high", patterns: [/\b(rap|bars|mic check|verse|rhyme scheme)\b/i] },
    { style: "instruct_archaic", label: "archaic wording", risk: "medium", patterns: [/\b(hark|thee|thou|beseech|pray tell|whence|henceforth)\b/i] },
    { style: "instruct_all_caps", label: "all-caps emphasis", risk: "medium", patterns: [/^[^a-z]*[A-Z][^a-z]*$/] },
    { style: "instruct_emoji", label: "emoji or emoticon", risk: "medium", patterns: [/[\u{1F300}-\u{1FAFF}]/u, /(:-\)|:\)|;\)|:-D|:D)/] },
    { style: "instruct_leet_speak", label: "leet or obfuscated text", risk: "high", patterns: [/\b[\w@#$]*[013457][\w@#$]*\b/i] },
    { style: "instruct_base64", label: "encoded text", risk: "high", patterns: [/\b[A-Za-z0-9+/]{32,}={0,2}\b/] },
    { style: "instruct_typo_random", label: "typos or keyboard noise", risk: "high", patterns: [/\b[a-z]*([a-z])\1{2,}[a-z]*\b/i, /\b(asdf|qwerty|jkl)\b/i, /[^\s]{18,}/] },
    { style: "instruct_bullet_list", label: "bullet-list format", risk: "low", patterns: [/\b(bullet|bulleted|list of|enumerate)\b/i] },
    { style: "instruct_numbered_steps", label: "numbered steps", risk: "low", patterns: [/\b(step by step|numbered steps|show steps|work through)\b/i] },
    { style: "instruct_output_json", label: "JSON output request", risk: "low", patterns: [/\b(json|object|schema|key-value)\b/i] },
    { style: "instruct_output_yaml", label: "YAML output request", risk: "low", patterns: [/\b(yaml)\b/i] },
    { style: "instruct_output_sql", label: "SQL output request", risk: "medium", patterns: [/\b(sql|select|insert into|table schema)\b/i] },
    { style: "instruct_code_fence", label: "code-fenced or markdown code", risk: "low", patterns: [/```/, /\b(code block|code fence)\b/i] },
    { style: "instruct_markdown_doc", label: "markdown document structure", risk: "low", patterns: [/\b(markdown|heading|section|table)\b/i] },
    { style: "instruct_with_citations", label: "citation request", risk: "medium", patterns: [/\b(cite|citation|sources|references|bibliography)\b/i] },
    { style: "instruct_with_summary", label: "summary request", risk: "low", patterns: [/\b(summary|summarize|tldr|tl;dr)\b/i] },
    { style: "instruct_risks_and_benefits", label: "risk-benefit framing", risk: "low", patterns: [/\b(risks? and benefits?|pros and cons|tradeoffs?)\b/i] },
    { style: "instruct_exact_numbers", label: "exact numeric constraints", risk: "low", patterns: [/\b(exactly|no more than|at least|top \d+|\d+ (items|examples|tips|steps))\b/i] },
    { style: "instruct_fewest_words", label: "extreme brevity constraint", risk: "high", patterns: [/\b(fewest words|one word|only \d+ words|ultra concise)\b/i] }
];

const TOPIC_FEATURES = {
    gsm8k: [
        { label: "arithmetic quantities", weight: 3, pattern: /\b(how many|total|altogether|left|remaining|each|per|twice|half|percent|ratio)\b/i },
        { label: "explicit numbers", weight: 2, pattern: /\d/ },
        { label: "math operations", weight: 3, pattern: /\b(calculate|solve|equation|sum|difference|product|divide|multiply)\b/i }
    ],
    mmlu: [
        { label: "moral scenario", weight: 4, pattern: /\b(moral|morally|wrong|ordinary standards|scenario 1|scenario 2|ethics|ethical)\b/i },
        { label: "knowledge discipline", weight: 2, pattern: /\b(philosophy|law|history|economics|biology|chemistry|physics|computer science|psychology)\b/i },
        { label: "multiple choice", weight: 2, pattern: /\b(which of these|choose|answer option|option [abcd])\b/i }
    ],
    alpaca: [
        { label: "open-ended instruction", weight: 2, pattern: /\b(write|generate|explain|describe|summarize|create|classify|translate|improve)\b/i },
        { label: "general advice or writing", weight: 2, pattern: /\b(tips|email|article|story|plan|list|rewrite|dialogue)\b/i }
    ]
};

const RISKY_STYLE_RE = /(leet|base64|morse|rot13|reversed|typo|fewest|haiku|poetic|rap|emoji|emoticon|empty_input|no_spaces|random|key_smash|malapropism)/;
const NON_RECOMMENDABLE_STYLE_RE = /(spoiler|curly_quotation|markdown_italic|markdown_bold|alternating_case|all_caps|lowercase|uppercase|klingon|pirate|yoda|shakespeare|archaic|sarcastic|profane|insulting)/;
const STABLE_STYLE_BONUS = {
    instruct_direct_question: 2.0,
    instruct_polite_request: 1.5,
    instruct_bullet_list: 1.3,
    instruct_numbered_steps: 1.4,
    instruct_with_step_by_step: 1.4,
    instruct_helpful_markdown_structure: 1.2,
    instruct_output_markdown: 1.0,
    instruct_exact_numbers: 1.0,
    instruct_short_paragraph: 0.8,
    instruct_with_summary: 0.8
};

function analyzeAndDisplaySearch(query) {
    const resultsContainer = document.getElementById('search-results-container');
    const examplesContainer = document.getElementById('example-suggestions-container');
    if (!query.trim()) {
        resultsContainer.innerHTML = '<p>Please enter a prompt to analyze.</p>';
        examplesContainer.innerHTML = '';
        return;
    }

    const analysis = analyzePrompt(query);
    const detectedData = state.aggregatedData[analysis.detectedStyle];
    const recommendations = recommendStyles(analysis).slice(0, 3);
    const examples = getRandomTestedExamples(
        recommendations.map(r => r.style),
        new Set(state.topStyleExampleKeys),
        3
    );

    renderExampleSuggestions(examples);

    const warning = analysis.topic.dataset !== state.currentDataset
        ? `<p class="warning">Detected topic looks closer to ${analysis.topic.dataset.toUpperCase()} than the selected ${state.currentDataset.toUpperCase()} dataset.</p>`
        : '';
    const detectedScore = detectedData
        ? `<span>TF ${detectedData.averages[0].toFixed(2)}</span><span>Overall ${detectedData.overallAverage.toFixed(2)}</span>`
        : `<span>No score data for detected style</span>`;
    const signalItems = analysis.signals.slice(0, 8).map(signal =>
        `<li><strong>${escapeHTML(signal.label)}</strong> (${signal.risk} risk)</li>`
    ).join('');
    const riskHtml = analysis.risks.length
        ? `<div class="analysis-panel warning-panel"><h4>Fragility Warnings</h4><ul>${analysis.risks.map(r => `<li>${escapeHTML(r)}</li>`).join('')}</ul></div>`
        : '';
    const nearestHtml = analysis.nearestPrompts.length
        ? `<div class="analysis-panel"><h4>Closest Tested Base Prompts</h4>${analysis.nearestPrompts.map(item => `
            <p class="nearest-prompt">${escapeHTML(trimText(item.prompt, 180))}</p>
        `).join('')}</div>`
        : '';
    const recommendationHtml = recommendations.map(rec => `
        <article class="recommendation-card">
            <h4>${formatParaphraseStyle(rec.style)}</h4>
            <div class="score-pills">
                <span>TF ${rec.data.averages[0].toFixed(2)}</span>
                <span>Overall ${rec.data.overallAverage.toFixed(2)}</span>
            </div>
            <p>${escapeHTML(rec.reason)}</p>
        </article>
    `).join('');

    resultsContainer.innerHTML = `
        <div class="analysis-grid">
            <div class="analysis-panel">
                <h4>Detected Style</h4>
                <p><strong>${formatParaphraseStyle(analysis.detectedStyle)}</strong></p>
                <div class="score-pills">${detectedScore}</div>
                ${signalItems ? `<ul>${signalItems}</ul>` : `<p>No strong paraphrase-style cue detected.</p>`}
            </div>
            <div class="analysis-panel">
                <h4>Detected Topic</h4>
                <p><strong>${analysis.topic.dataset.toUpperCase()}</strong></p>
                <p>${escapeHTML(analysis.topic.evidence.join(', ') || 'general instruction-following')}</p>
                ${warning}
            </div>
            ${riskHtml}
            ${nearestHtml}
        </div>
        <div class="recommendation-grid">${recommendationHtml}</div>
    `;
}

function analyzePrompt(query) {
    const tokens = tokenize(query);
    const styleScores = new Map();
    const signals = [];
    STYLE_FEATURES.forEach(feature => {
        const matches = feature.patterns.reduce((sum, pattern) => sum + countMatches(query, pattern), 0);
        if (matches > 0) {
            styleScores.set(feature.style, (styleScores.get(feature.style) || 0) + matches);
            signals.push({ ...feature, matches });
        }
    });

    const detectedStyle = Array.from(styleScores.entries())
        .sort((a, b) => b[1] - a[1])[0]?.[0] || "instruction_original";
    const topic = classifyTopic(query);
    const nearestPrompts = findNearestPrompts(tokens);
    const risks = collectRiskWarnings(query, signals, detectedStyle);
    return { query, tokens, detectedStyle, styleScores, signals, topic, nearestPrompts, risks };
}

function classifyTopic(query) {
    const scores = { alpaca: 0, gsm8k: 0, mmlu: 0 };
    const evidence = { alpaca: [], gsm8k: [], mmlu: [] };
    Object.entries(TOPIC_FEATURES).forEach(([dataset, features]) => {
        features.forEach(feature => {
            if (feature.pattern.test(query)) {
                scores[dataset] += feature.weight;
                evidence[dataset].push(feature.label);
            }
        });
    });
    const dataset = Object.entries(scores).sort((a, b) => b[1] - a[1])[0][0];
    return {
        dataset: scores[dataset] === 0 ? state.currentDataset : dataset,
        scores,
        evidence: evidence[dataset]
    };
}

function collectRiskWarnings(query, signals, detectedStyle) {
    const warnings = [];
    signals.filter(signal => signal.risk === "high").forEach(signal => {
        warnings.push(`${signal.label} is a high-fragility CALIPER cue for several model/dataset settings.`);
    });
    if (query.length > 700) warnings.push("Very long prompts can hide the core task; keep the instruction and output format easy to locate.");
    if (tokenize(query).length < 5) warnings.push("Very short prompts can omit constraints that models need for reliable task completion.");
    const data = state.aggregatedData[detectedStyle];
    if (data && data.averages[0] < 6) {
        warnings.push(`${formatParaphraseStyle(detectedStyle)} has low Task Fulfilment for the selected dataset/model.`);
    }
    return warnings;
}

function recommendStyles(analysis) {
    return getRecommendableStyles().map(([style, data]) => {
        let score = data.averages[0] * 1.8 + data.overallAverage * 0.6 + (STABLE_STYLE_BONUS[style] || 0);
        if (analysis.topic.dataset === "gsm8k" && /(step|number|exact|markdown|direct)/.test(style)) score += 1.2;
        if (analysis.topic.dataset === "mmlu" && /(direct|neutral|summary|risk|markdown)/.test(style)) score += 1.0;
        if (analysis.topic.dataset === "alpaca" && /(bullet|markdown|summary|direct|polite)/.test(style)) score += 0.8;
        if (analysis.styleScores.has(style)) score += 0.4;
        const reason = buildRecommendationReason(style, analysis.topic.dataset);
        return { style, data, score, reason };
    }).filter(Boolean).sort((a, b) => b.score - a.score);
}

function buildRecommendationReason(style, dataset) {
    if (/step|numbered|exact/.test(style)) return "Keeps intermediate constraints explicit, which is useful for structured tasks.";
    if (/bullet|markdown|summary/.test(style)) return "Keeps the request scannable without obscuring the task.";
    if (/direct|polite/.test(style)) return "Preserves the core request with low stylistic overhead.";
    if (/risk/.test(style)) return "Makes the judgment criteria explicit while retaining the original task.";
    return `Ranks well for ${dataset.toUpperCase()} on the selected model while avoiding common obfuscation and typo failure modes.`;
}

function renderExampleSuggestions(examples) {
    const container = document.getElementById('example-suggestions-container');
    if (!examples.length) {
        container.innerHTML = '';
        return;
    }
    container.innerHTML = `
        <div class="tested-examples">
            <h3>Tested Examples To Try</h3>
            <div class="example-grid">
                ${examples.map(example => `
                    <button class="example-card" type="button" data-prompt="${escapeAttribute(example.prompt)}">
                        <span>${formatParaphraseStyle(example.style)}</span>
                        <strong>TF ${example.tf.toFixed(1)}</strong>
                        <p>${escapeHTML(trimText(example.prompt, 220))}</p>
                    </button>
                `).join('')}
            </div>
        </div>
    `;
    container.querySelectorAll('.example-card').forEach(card => {
        card.addEventListener('click', () => {
            document.getElementById('search-input').value = card.dataset.prompt;
            analyzeAndDisplaySearch(card.dataset.prompt);
        });
    });
}

function getRandomTestedExamples(styles, excludedKeys, count) {
    const preferred = state.examples.filter(example =>
        styles.includes(example.style) && !excludedKeys.has(getExampleKey(example))
    );
    const fallback = state.examples.filter(example => !excludedKeys.has(getExampleKey(example)));
    const candidates = preferred.length >= count ? preferred : preferred.concat(fallback);
    const shuffled = candidates
        .map(example => ({ example, sort: Math.random() }))
        .sort((a, b) => a.sort - b.sort)
        .map(item => item.example);
    const selected = [];
    const used = new Set(excludedKeys);
    for (const example of shuffled) {
        const key = getExampleKey(example);
        if (used.has(key)) continue;
        selected.push(example);
        used.add(key);
        if (selected.length === count) break;
    }
    return selected;
}

function findExampleForStyle(style, excludedKeys = new Set()) {
    const scoredExample = state.examples
        .filter(example => example.style === style && !excludedKeys.has(getExampleKey(example)))
        .sort((a, b) => (b.tf - a.tf) || (b.average - a.average))[0];
    if (scoredExample) return scoredExample;

    return {
        dataset: state.currentDataset,
        model: state.currentModel,
        prompt_count: `style-example-${style}`,
        style,
        prompt: getStyleTooltip(style),
    };
}

function isRecommendableStyle(style) {
    return style !== 'instruction_original'
        && !RISKY_STYLE_RE.test(style)
        && !NON_RECOMMENDABLE_STYLE_RE.test(style);
}

function getRecommendableStyles(metricIndex = 0) {
    return Object.entries(state.aggregatedData)
        .filter(([style, data]) => isRecommendableStyle(style) && data.count > 0)
        .map(([style, data]) => {
            const score = data.averages[metricIndex] * 1.5
                + data.overallAverage * 0.7
                + (STABLE_STYLE_BONUS[style] || 0);
            return [style, data, score];
        })
        .sort((a, b) => (b[2] - a[2]) || (b[1].averages[metricIndex] - a[1].averages[metricIndex]))
        .map(([style, data]) => [style, data]);
}

function getRankedStyles(metricIndex = 0) {
    return Object.entries(state.aggregatedData)
        .filter(([, data]) => data.count > 0)
        .sort(([, a], [, b]) => (b.averages[metricIndex] - a.averages[metricIndex]) || (b.overallAverage - a.overallAverage));
}

function buildCorpusSignals() {
    const termFreq = new Map();
    const docs = state.instructions.map(item => {
        const prompt = getPromptText(item);
        const tokens = new Set(tokenize(prompt).filter(token => !STOPWORDS.has(token) && token.length > 2));
        tokens.forEach(token => termFreq.set(token, (termFreq.get(token) || 0) + 1));
        return { key: getPromptKey(item), prompt, tokens };
    });
    return { termFreq, docs };
}

function findNearestPrompts(queryTokens) {
    const querySet = new Set(queryTokens.filter(token => !STOPWORDS.has(token) && token.length > 2));
    if (!querySet.size || !state.corpusSignals.docs) return [];
    return state.corpusSignals.docs.map(doc => {
        let score = 0;
        querySet.forEach(token => {
            if (doc.tokens.has(token)) {
                const freq = state.corpusSignals.termFreq.get(token) || 1;
                score += 1 / Math.sqrt(freq);
            }
        });
        return { ...doc, score };
    }).filter(item => item.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 3);
}

function tokenize(text) {
    return (text.toLowerCase().match(/[a-z0-9]+(?:'[a-z]+)?/g) || []);
}

function countMatches(text, pattern) {
    const flags = pattern.flags.includes('g') ? pattern.flags : `${pattern.flags}g`;
    return (text.match(new RegExp(pattern.source, flags)) || []).length;
}

function getPromptKey(item) {
    return item?.prompt_count ?? item?.prompt_id ?? item?.id ?? null;
}

function getPromptText(item) {
    return item?.instruction_original || item?.instruction || item?.prompt || "";
}

function getExampleKey(example) {
    return `${example.dataset}|${example.model}|${example.prompt_count}|${example.style}`;
}

function getStyleTooltip(style) {
    return state.styleTooltipExamples[style]
        || state.styleTooltipExamples.instruction_original
        || "What are the three primary colors?";
}

function trimText(text, maxLength) {
    if (!text || text.length <= maxLength) return text || "";
    return `${text.slice(0, maxLength - 1).trim()}...`;
}

function escapeHTML(value) {
    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

function escapeAttribute(value) {
    return escapeHTML(value).replace(/`/g, "&#96;");
}



function calculateAverage(arr) {
    if (!arr || arr.length === 0) return 0;
    const sum = arr.reduce((acc, val) => acc + val, 0);
    return sum / arr.length;
}

function calculateStdDev(arr, mean) {
    if (!arr || arr.length < 2) return 0;
    const avg = mean !== undefined ? mean : calculateAverage(arr);
    const squareDiffs = arr.map(value => Math.pow(value - avg, 2));
    const avgSquareDiff = calculateAverage(squareDiffs);
    return Math.sqrt(avgSquareDiff);
}


function scoreToColor(score, opacity = 0.45) {
  score = Math.max(0, Math.min(10, score));
  let lightness, saturation;

  if (score <= 7.3) {
    const t = score / 7.3;
    lightness  = 95 - 5 * t;
    saturation = 20 * t;
  } else if (score <= 8.4) {
    const t = (score - 7.3) / 1.1;
    lightness  = 90 - 45 * t;
    saturation = 20 + 70 * t;
  } else {
    const t = (score - 8.4) / 1.6;
    lightness  = 45 - 10 * t;
    saturation = 90 + 10 * t;
  }

  return `hsla(120, ${saturation}%, ${lightness}%, ${opacity})`;
}
