// ==UserScript==
// @name         BOSS直聘-AI标注 (v1.9 - 性能优化版)
// @namespace    http://tampermonkey.net/
// @version      1.9
// @description  【性能优化】引入Debounce(防抖)机制，大幅降低CPU占用，彻底解决页面卡顿问题，提升流畅度。
// @author       Your Name & AI Assistant
// @match        *://www.zhipin.com/web/geek/*
// @match        *://www.zhipin.com/web/chat/*
// @match        *://www.zhipin.com/web/frame/recommend*
// @connect      127.0.0.1
// @grant        GM_xmlhttpRequest
// @grant        GM_addStyle
// ==/UserScript==

(function() {
    'use strict';

    // --- 配置区 ---
    const API_ENDPOINT = 'http://127.0.0.1:5002/get_candidate_info?name=';
    const NAME_SELECTOR = 'span.name, .geek-name';
    const POLLING_INTERVAL = 3000; // 轮询间隔 (毫秒)
    const DEBOUNCE_DELAY = 300;   // 防抖延迟 (毫秒)

    // --- 【【【 新增的性能优化核心：防抖函数 】】】 ---
    function debounce(func, wait) {
        let timeout;
        return function(...args) {
            const context = this;
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(context, args), wait);
        };
    }

    // --- 样式注入 ---
    GM_addStyle(`
        .ai-assistant-tag { display: inline-block; padding: 3px 8px; margin-left: 12px; border-radius: 5px; font-size: 12px; font-weight: bold; color: white !important; vertical-align: middle; cursor: help; transition: all 0.2s ease; z-index: 9999; }
        .ai-assistant-tag:hover { transform: scale(1.05); box-shadow: 0 0 10px rgba(0,0,0,0.2); }
        .ai-tag-high { background-color: #28a745; }
        .ai-tag-medium { background-color: #ffc107; color: black !important; }
        .ai-tag-low { background-color: #dc3545; }
        .ai-tag-error { background-color: #6c757d; }
    `);

    // --- 核心逻辑 (保持不变) ---
    function fetchAIData(name) {
        return new Promise((resolve) => {
            GM_xmlhttpRequest({
                method: 'GET',
                url: API_ENDPOINT + encodeURIComponent(name),
                onload: (response) => {
                    if (response.status === 200) { try { resolve(JSON.parse(response.responseText)); } catch (e) { resolve(null); } }
                    else { resolve(null); }
                },
                onerror: () => resolve(null)
            });
        });
    }

    function getTagClassByScore(score) {
        if (score >= 8.5) return 'ai-tag-high';
        if (score >= 6.5) return 'ai-tag-medium';
        return 'ai-tag-low';
    }

    function createOrUpdateTag(nameElement, data) {
        if (!nameElement || !nameElement.parentNode) return;
        let tag = nameElement.parentNode.querySelector('.ai-assistant-tag');
        if (!tag) {
            tag = document.createElement('span');
            tag.className = 'ai-assistant-tag';
            nameElement.insertAdjacentElement('afterend', tag);
        }
        if (data && data.name) {
            const score = data.score || 0.0;
            const recommendation = data.recommendation || '';
            const recMatch = recommendation.match(/【([^】]+)】/);
            tag.className = 'ai-assistant-tag ' + getTagClassByScore(score);
            tag.textContent = `AI: ${score.toFixed(1)}分 | ${recMatch ? recMatch[1] : '查看'}`;
            tag.title = `最匹配职位: ${data.best_position || 'N/A'}\n市场竞争力: ${data.market_competitiveness || 'N/A'}\n\n完整建议: ${recommendation}`;
        } else {
            tag.className = 'ai-assistant-tag ai-tag-error';
            tag.textContent = 'AI: 未分析';
            tag.title = '该候选人尚未被本地AI助理分析处理。';
        }
    }

    function processInitialCandidates() {
        document.querySelectorAll(NAME_SELECTOR + ':not([data-ai-processed="true"])').forEach(async (el) => {
            el.setAttribute('data-ai-processed', 'true');
            const name = el.textContent.trim();
            if (name) {
                const aiData = await fetchAIData(name);
                createOrUpdateTag(el, aiData);
            }
        });
    }

    async function updatePendingTags() {
        const pendingTags = document.querySelectorAll('.ai-tag-error');
        if (pendingTags.length === 0) return;
        for (const tag of pendingTags) {
            const nameElement = tag.previousElementSibling;
            if (nameElement && nameElement.matches(NAME_SELECTOR)) {
                const name = nameElement.textContent.trim();
                if (name) {
                    const aiData = await fetchAIData(name);
                    if (aiData && aiData.name) {
                        createOrUpdateTag(nameElement, aiData);
                    }
                }
            }
        }
    }

    // --- 启动逻辑 ---
    console.log(`[AI助理脚本 v1.9 - 性能优化版] 启动成功！`);

    // 1. 创建一个经过“防抖”处理的函数
    const debouncedProcess = debounce(processInitialCandidates, DEBOUNCE_DELAY);

    // 2. 让 MutationObserver 调用这个“防抖”后的函数
    const observer = new MutationObserver(() => {
        debouncedProcess();
    });
    observer.observe(document.body, { childList: true, subtree: true });

    // 3. 页面初次加载时，处理已存在的候选人
    processInitialCandidates();

    // 4. 启动轮询巡查员 (保持不变)
    setInterval(updatePendingTags, POLLING_INTERVAL);

})();








// ==UserScript==
// @name         【猎聘专用】AI 招聘助理 (v7.1 - 最终精简优化版)
// @namespace    http://tampermonkey.net/
// @version      7.1
// @description  【终极优化】精简选择器，解决沟通列表因“双层姓名”结构导致的显示不全或重复问题。
// @author       Your Name & AI Assistant
// @match        *://*.liepin.com/resume/*
// @match        *://h.liepin.com/*
// @connect      127.0.0.1
// @grant        GM_xmlhttpRequest
// @grant        GM_addStyle
// ==/UserScript==

(function() {
    'use strict';

    // --- 配置区 ---
    const API_ENDPOINT = 'http://127.0.0.1:5003/get_candidate_info?name=';
    const NAME_SELECTOR = [
        'p.name-title',
        '.text-overflow-1.name',
        '.name-box > .name-text',
        'div.name-wrapper > span.name',
        'div.new-resume-personal-name > em',
        'h1.name',
        'span.__im_basic__contact-title-name',
        // 'span.__im_pro__contact-title-title', // <--- 【【【 核心优化：注释掉这个多余的选择器 】】】
        'div.__im_basic__user-title > button',
        'h4.name.ellipsis',
        // --- 【【【 本次核心新增：只抓取沟通列表里的真实姓名 】】】 ---
        'span.__im_pro__contact-title-name'
    ].join(', ');

    const POLLING_INTERVAL = 2000;
    const DEBOUNCE_DELAY = 300;

    // --- (后面的所有代码，从样式注入到启动逻辑，都与 v7.0 完全相同，无需改动) ---
    function debounce(func, wait) { let timeout; return function(...args) { const context = this; clearTimeout(timeout); timeout = setTimeout(() => func.apply(context, args), wait); }; }
    GM_addStyle(`
        .ai-assistant-tag { display: inline-block; padding: 3px 8px; margin-left: 10px; border-radius: 5px; font-size: 12px; font-weight: bold; color: white !important; vertical-align: middle; cursor: help; transition: all 0.2s ease; z-index: 9999; }
        .ai-assistant-tag:hover { transform: scale(1.05); box-shadow: 0 0 10px rgba(0,0,0,0.2); }
        .ai-tag-high { background-color: #28a745; }
        .ai-tag-medium { background-color: #ffc107; color: black !important; }
        .ai-tag-low { background-color: #dc3545; }
        .ai-tag-error { background-color: #6c757d; }
    `);
    function fetchAIData(name) { return new Promise((resolve) => { if (!name || name.trim() === '') return resolve(null); GM_xmlhttpRequest({ method: 'GET', url: API_ENDPOINT + encodeURIComponent(name), onload: (response) => { if (response.status === 200) { try { resolve(JSON.parse(response.responseText)); } catch (e) { resolve(null); } } else { resolve(null); } }, onerror: () => resolve(null) }); }); }
    function getTagClassByScore(score) { if (score >= 8.5) return 'ai-tag-high'; if (score >= 6.5) return 'ai-tag-medium'; return 'ai-tag-low'; }
    function createOrUpdateTag(nameElement, data) {
        if (!nameElement || !nameElement.isConnected) return;
        let tag = nameElement.nextElementSibling;
        if (!tag || !tag.classList.contains('ai-assistant-tag')) {
            tag = document.createElement('span');
            tag.className = 'ai-assistant-tag';
            nameElement.insertAdjacentElement('afterend', tag);
        }
        if (data && data.name) {
            const score = data.score || 0.0;
            const recommendation = data.recommendation || '';
            const recMatch = recommendation.match(/【([^】]+)】/);
            tag.className = 'ai-assistant-tag ' + getTagClassByScore(score);
            tag.textContent = `AI: ${score.toFixed(1)}分 | ${recMatch ? recMatch[1] : '查看'}`;
            tag.title = `最匹配职位: ${data.best_position || 'N/A'}\n市场竞争力: ${data.market_competitiveness || 'N/A'}\n\n完整建议: ${recommendation}`;
        } else {
            tag.className = 'ai-assistant-tag ai-tag-error';
            tag.textContent = 'AI: 未分析';
            tag.title = '该候选人尚未被本地AI助理分析处理。';
        }
    }
    function processNewCandidates() {
        document.querySelectorAll(NAME_SELECTOR + ':not([data-ai-processed="true"])').forEach(async (el) => {
            el.setAttribute('data-ai-processed', 'true');
            const name = el.textContent.trim();
            if (name) {
                const aiData = await fetchAIData(name);
                if (el.isConnected) { createOrUpdateTag(el, aiData); }
            }
        });
    }
    async function updatePendingTags() {
        for (const tag of document.querySelectorAll('.ai-tag-error')) {
            const nameElement = tag.previousElementSibling;
            if (nameElement && nameElement.matches(NAME_SELECTOR)) {
                const name = nameElement.textContent.trim();
                if (name) {
                    const aiData = await fetchAIData(name);
                    if (aiData && aiData.name) {
                        if (nameElement.isConnected) { createOrUpdateTag(nameElement, aiData); }
                    }
                }
            }
        }
    }
    console.log(`[AI助理脚本 v7.1 - 最终精简优化版] 启动成功！`);
    const debouncedProcess = debounce(processNewCandidates, DEBOUNCE_DELAY);
    const observer = new MutationObserver(debouncedProcess);
    observer.observe(document.body, { childList: true, subtree: true });
    processNewCandidates();
    setInterval(updatePendingTags, POLLING_INTERVAL);

})();