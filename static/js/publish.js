/* ==========================================================================
   发布页：图片上传（点击/拖拽/粘贴）、标签输入、提交并展示即时匹配结果
   ========================================================================== */
'use strict';

(function () {
  const MAX_IMAGES = 4;
  const form = document.getElementById('publish-form');
  if (!form) return;

  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const previewGrid = document.getElementById('preview-grid');
  const tagInputWrap = document.getElementById('tag-input');
  const tagChips = document.getElementById('tag-chips');
  const tagField = document.getElementById('f-tags');
  const tagSuggest = document.getElementById('tag-suggest');
  const submitBtn = document.getElementById('submit-btn');
  const formStatus = document.getElementById('form-status');
  const resultPanel = document.getElementById('result-panel');
  const resultTitle = document.getElementById('result-title');
  const resultMatches = document.getElementById('result-matches');
  const resultLink = document.getElementById('result-link');

  /** @type {File[]} 待上传的图片 */
  let files = [];
  /** @type {string[]} 已录入的标签 */
  let tags = [];

  const SUGGESTED = [
    '黑色', '白色', '蓝色', '粉色', '绿色', '红色', '棕色',
    '双肩包', '书包', '钱包', '耳机', '手机', '校园卡', '一卡通', '钥匙',
    '水杯', '保温杯', '雨伞', '眼镜', '手表', '手环', '笔记本', '电脑', '平板',
    '图书馆', '食堂', '教学楼', '宿舍', '操场', '体育馆', '实验楼', '校车',
    '贴纸', '挂件', '划痕', '贴膜', '钥匙扣', '充电器',
  ];

  /* ---------------------------------------------------------------- 图片 */
  function renderPreviews() {
    previewGrid.innerHTML = '';
    files.forEach((file, index) => {
      const url = URL.createObjectURL(file);
      const item = document.createElement('div');
      item.className = 'preview-item';
      item.innerHTML = `<img src="${url}" alt="预览 ${index + 1}">
        <button type="button" class="preview-remove" title="移除" data-index="${index}">×</button>`;
      previewGrid.appendChild(item);
    });
    previewGrid.querySelectorAll('.preview-remove').forEach((btn) => {
      btn.addEventListener('click', () => {
        files.splice(Number(btn.dataset.index), 1);
        renderPreviews();
      });
    });
  }

  function addFiles(list) {
    const incoming = Array.from(list || []).filter((f) => f.type.startsWith('image/'));
    if (!incoming.length) return;
    const room = MAX_IMAGES - files.length;
    if (room <= 0) { LF.toast(`最多上传 ${MAX_IMAGES} 张照片`, 'error'); return; }
    if (incoming.length > room) LF.toast(`最多 ${MAX_IMAGES} 张，已只保留前 ${room} 张`, 'error');
    files = files.concat(incoming.slice(0, room));
    renderPreviews();
  }

  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener('change', () => { addFiles(fileInput.files); fileInput.value = ''; });

  ['dragenter', 'dragover'].forEach((type) => {
    dropzone.addEventListener(type, (event) => { event.preventDefault(); dropzone.classList.add('dragover'); });
  });
  ['dragleave', 'drop'].forEach((type) => {
    dropzone.addEventListener(type, (event) => { event.preventDefault(); dropzone.classList.remove('dragover'); });
  });
  dropzone.addEventListener('drop', (event) => addFiles(event.dataTransfer.files));

  // 直接粘贴截图/图片
  document.addEventListener('paste', (event) => {
    const items = event.clipboardData && event.clipboardData.items;
    if (!items) return;
    const pasted = [];
    for (const item of items) {
      if (item.kind === 'file' && item.type.startsWith('image/')) {
        const file = item.getAsFile();
        if (file) pasted.push(file);
      }
    }
    if (pasted.length) {
      addFiles(pasted);
      LF.toast(`已粘贴 ${pasted.length} 张图片`, 'ok');
    }
  });

  /* ---------------------------------------------------------------- 标签 */
  function renderTags() {
    tagChips.innerHTML = tags
      .map((tag, index) => `<span class="tag-chip">${LF.esc(tag)}<button type="button" data-index="${index}" title="移除">×</button></span>`)
      .join('');
    tagChips.querySelectorAll('button').forEach((btn) => {
      btn.addEventListener('click', () => {
        tags.splice(Number(btn.dataset.index), 1);
        renderTags();
      });
    });
  }

  function addTag(raw) {
    const value = String(raw || '').trim().replace(/^[#,，、]+|[#,，、]+$/g, '');
    if (!value) return;
    if (value.length > 24) { LF.toast('单个标签不超过 24 个字', 'error'); return; }
    if (tags.includes(value)) return;
    if (tags.length >= 16) { LF.toast('最多 16 个标签', 'error'); return; }
    tags.push(value);
    renderTags();
  }

  tagField.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ',' || event.key === '，' || event.key === '、') {
      event.preventDefault();
      addTag(tagField.value);
      tagField.value = '';
    } else if (event.key === 'Backspace' && !tagField.value && tags.length) {
      tags.pop();
      renderTags();
    }
  });
  // 失焦时也把未确认的输入并入标签，避免用户以为已添加
  tagField.addEventListener('blur', () => {
    if (tagField.value.trim()) { addTag(tagField.value); tagField.value = ''; }
  });
  tagInputWrap.addEventListener('click', () => tagField.focus());

  tagSuggest.innerHTML = SUGGESTED.map((tag) => `<button type="button" data-tag="${LF.esc(tag)}">+ ${LF.esc(tag)}</button>`).join('');
  tagSuggest.querySelectorAll('button').forEach((btn) => {
    btn.addEventListener('click', () => addTag(btn.dataset.tag));
  });

  /* ---------------------------------------------------------------- 提交 */
  form.addEventListener('submit', async (event) => {
    event.preventDefault();

    const title = document.getElementById('f-title').value.trim();
    if (title.length < 2) { LF.toast('请填写标题（至少 2 个字）', 'error'); return; }
    if (tagField.value.trim()) { addTag(tagField.value); tagField.value = ''; }

    const typeInput = form.querySelector('input[name="type"]:checked');
    const formData = new FormData();
    formData.append('type', typeInput ? typeInput.value : 'lost');
    formData.append('title', title);
    formData.append('category', document.getElementById('f-category').value);
    formData.append('color_name', document.getElementById('f-color').value.trim());
    formData.append('location', document.getElementById('f-location').value.trim());
    formData.append('description', document.getElementById('f-desc').value.trim());
    formData.append('contact_name', document.getElementById('f-name').value.trim());
    formData.append('contact', document.getElementById('f-contact').value.trim());
    formData.append('tags', tags.join(','));
    files.forEach((file) => formData.append('images', file, file.name));

    submitBtn.disabled = true;
    submitBtn.textContent = '正在上传并匹配…';
    formStatus.textContent = files.length
      ? `上传 ${files.length} 张照片并提取感知特征…`
      : '未上传照片，将仅使用文本标签通道匹配…';

    try {
      const data = await LF.api('/api/items', { method: 'POST', formData });
      const item = data.item || {};
      resultPanel.hidden = false;
      resultTitle.textContent = `发布成功：${item.title || ''}`;
      resultLink.href = `/item/${item.id}`;
      resultMatches.innerHTML = LF.renderMatches(
        data.matches,
        `<div class="empty-state small"><h3>暂时没有找到相似记录</h3>
         <p>库内还没有与之相似的对侧记录。新记录已入库，之后若有新发布，系统会自动重新撮合。</p></div>`
      );
      formStatus.textContent = `匹配到 ${data.match_count || 0} 条候选`;
      LF.toast(`发布成功，匹配到 ${data.match_count || 0} 条候选`, 'ok');
      resultPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });

      // 重置表单，方便连续发布
      form.reset();
      tags = [];
      files = [];
      renderTags();
      renderPreviews();
    } catch (err) {
      formStatus.textContent = '';
      LF.toast(err.message, 'error');
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = '发布并立即匹配';
    }
  });
})();
