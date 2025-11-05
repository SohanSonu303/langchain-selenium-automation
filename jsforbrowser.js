(function() {
    // Start performance timer to see how long the scraping takes.
    console.time("EnhancedElementScraping");

    /**
     * Generates a robust XPath for a given DOM element.
     * Prefers using a unique ID if available, otherwise constructs a path from the root.
     * @param {Element} element The DOM element to generate the XPath for.
     * @returns {string} The XPath string.
     */
    const getXPathForElement = (element) => {
        if (element && element.id) {
            return `//*[@id='${element.id}']`;
        }
        const parts = [];
        while (element && element.nodeType === Node.ELEMENT_NODE) {
            let nbOfPreviousSiblings = 0;
            let hasNextSiblings = false;
            let sibling = element.previousSibling;
            while (sibling) {
                if (sibling.nodeType === Node.ELEMENT_NODE && sibling.nodeName === element.nodeName) {
                    nbOfPreviousSiblings++;
                }
                sibling = sibling.previousSibling;
            }
            sibling = element.nextSibling;
            while (sibling) {
                if (sibling.nodeName === element.nodeName) {
                    hasNextSiblings = true;
                    break;
                }
                sibling = sibling.nextSibling;
            }
            const tagName = element.nodeName.toLowerCase();
            const part = (nbOfPreviousSiblings || hasNextSiblings) ?
                `${tagName}[${nbOfPreviousSiblings + 1}]` :
                tagName;
            parts.unshift(part);
            element = element.parentNode;
        }
        return parts.length ? '/' + parts.join('/') : null;
    };

    /**
     * Gets contextual information from an element's ancestors.
     * Captures the immediate parent and the closest form ancestor for better context.
     * @param {Element} element The element to analyze.
     * @returns {object | null} An object containing contextual info, or null if no parent.
     */
    const getAncestorContext = (element) => {
        let parent = element.parentElement;
        if (parent) {
            const parentContext = {
                tagName: parent.tagName.toLowerCase(),
                id: parent.id || null,
                role: parent.getAttribute('role') || null,
                ariaLabel: parent.getAttribute('aria-label') || null,
            };
            
            // Try to find a more meaningful ancestor like a form
            let formAncestor = element.closest('form');
            if (formAncestor) {
                 parentContext.form = { id: formAncestor.id || null, name: formAncestor.name || null };
            }
            return parentContext;
        }
        return null;
    }

    // A comprehensive selector for elements a user is likely to interact with.
    const interactiveSelector = 'a, button, input, select, textarea, label, h1, h2, h3, h4, [role="button"], [role="link"], [role="tab"], [onclick]';
    const allPotentialElements = document.querySelectorAll(interactiveSelector);
    const elementDetails = [];
    const processedElements = new Set(); // To avoid processing the same element twice

    allPotentialElements.forEach(element => {
        // Skip if this element has already been processed (e.g., a label containing an input)
        if (processedElements.has(element)) {
            return;
        }

        // --- FILTERING RULES TO IGNORE IRRELEVANT ELEMENTS ---
        const style = getComputedStyle(element);
        const rect = element.getBoundingClientRect();

        // Rule 1: Skip invisible elements.
        if (rect.width === 0 || rect.height === 0 || element.offsetParent === null || style.visibility === 'hidden' || style.display === 'none') {
            return;
        }

        // --- DATA EXTRACTION ---

        // Part 1: Extract various forms of text for identification
        const mainText = (element.textContent || element.innerText || "").trim();
        const valueText = element.value ? element.value.trim() : "";
        const visibleText = mainText || valueText;
        
        // Find associated label text, which is often the best identifier for form elements
        let labelText = '';
        if (element.id) {
            const label = document.querySelector(`label[for="${element.id}"]`);
            if (label) {
                labelText = (label.textContent || label.innerText).trim();
            }
        }
        if (!labelText) { // Fallback to finding a parent label
            const parentLabel = element.closest('label');
            if (parentLabel) {
                 labelText = (parentLabel.textContent || parentLabel.innerText).trim();
            }
        }
        
        const ariaLabel = element.getAttribute('aria-label');
        const placeholder = element.getAttribute('placeholder');
        const name = element.getAttribute('name');
        
        // Combine all text sources to find the most likely identifier for the AI
        const computedText = (ariaLabel || labelText || visibleText || placeholder || name || "").replace(/\s+/g, ' ').trim();
        
        // Rule 2: Skip elements that have no discernible text or ID for identification.
        if (!element.id && !computedText) {
            return;
        }

        // If we've reached here, the element is considered meaningful. Mark as processed.
        processedElements.add(element);

        // Part 2: Gather all relevant details into a structured object
        elementDetails.push({
            tagName: element.tagName.toLowerCase(),
            attributes: {
                id: element.id || null,
                class: element.className || null,
                name: name || null,
                // Add the input 'type' attribute, which is semantically very important
                type: element.tagName.toLowerCase() === 'input' ? element.type : null,
                role: element.getAttribute('role') || null,
                ariaLabel: ariaLabel || null,
                placeholder: placeholder || null,
                href: element.getAttribute('href') || null,
            },
            // The 'state' object is crucial for an AI agent to determine if an action is possible
            state: {
                isDisabled: element.disabled,
                isReadOnly: element.readOnly,
                isChecked: element.checked, // For checkboxes/radio buttons
                isSelected: element.selected, // For options in a <select>
                isHiddenByAria: element.getAttribute('aria-hidden') === 'true',
            },
            // The 'text' object provides different text contexts for better matching
            text: {
                visibleText: visibleText || null, // The direct text on the element
                labelText: labelText || null, // Text from an associated <label>
                computedText: computedText, // The AI's best guess for the element's name
            },
            // The 'context' object helps understand the element's place on the page
            context: getAncestorContext(element),
            xpath: getXPathForElement(element),
            location: {
                x: rect.x,
                y: rect.y,
                width: rect.width,
                height: rect.height
            }
        });
    });
    
    // --- FINAL OUTPUT ---
    
    // Stop the performance timer
    console.timeEnd("EnhancedElementScraping");
    
    // Print a summary message in the console
    console.log(`%cSuccessfully scraped ${elementDetails.length} meaningful elements with enhanced context.`, "color: #4CAF50; font-weight: bold; font-size: 14px;");
    console.log("Use the table below for easy inspection or copy the complete JSON object for your application.");
    
    // Output 1: An interactive table in the console for easy browsing and sorting
    console.table(elementDetails);
    
    // Output 2: The full data as a clean JSON string, which can be easily copied
    console.log(JSON.stringify(elementDetails, null, 2));

})();
