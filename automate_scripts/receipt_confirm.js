/**
 * @fileoverview Sends a confirmation email with expense request details
 * to the person who submits an OChEGS expense request form.
 * @author OChEGS Treasurer
 * @version 2.0
 */

//================================================================
// UTILITY HELPER
//================================================================
/**
 * Safely retrieves a value from the form response object.
 * Google Forms sends even single responses as an array.
 *
 * @param {Object} namedValues The event object `e.namedValues` from the form submission.
 * @param {string} key The exact name of the form question/column header.
 * @param {string} [defaultValue='N/A'] The value to return if the key doesn't exist or is empty.
 * @returns {string} The response value or the default value.
 */
function getResponseValue(namedValues, key, defaultValue = 'N/A') {
  // Check if the key exists and the array has a value
  if (namedValues && namedValues[key] && namedValues[key][0]) {
    return namedValues[key][0];
  }
  return defaultValue;
}


//================================================================
// MAIN CONTROLLER FUNCTION
//================================================================
/**
 * The main function triggered by a Google Form submission.
 * It parses the response and triggers the confirmation email.
 *
 * @param {GoogleAppsScript.Events.FormsOnFormSubmit} e The event object passed by the trigger.
 */
function onFormSubmit(e) {
  try {
    // e.namedValues is the object containing {question: [answer]} pairs
    const { namedValues } = e;
    if (!namedValues) {
      console.log("This function should be run via a form submit trigger, not manually.");
      return;
    }

    // Using our helper to safely extract all data
    const submissionData = {
      fullName:        getResponseValue(namedValues, 'Full name', 'Requester'),
      email:           getResponseValue(namedValues, 'Email', ''),
      transactionType: getResponseValue(namedValues, 'Transaction type'),
      dateOfExpense:   getResponseValue(namedValues, 'Date of purchase/expense '),
      dateNeeded:      getResponseValue(namedValues, 'When do you need the funds?'),
      category:        getResponseValue(namedValues, 'Expense category'),
      description:     getResponseValue(namedValues, 'Description of expense', 'No description provided.'),
      amount:          getResponseValue(namedValues, 'Amount (USD)'),
      receiptLink:     getResponseValue(namedValues, 'Upload receipt (Receipts MUST be uploaded for reimbursement requests)', 'No receipt uploaded.'),
      paymentWorks:    getResponseValue(namedValues, 'UH Paymentworks Email', ''),
      otherComments:   getResponseValue(namedValues, 'Other comments', '') // Default to empty string
    };


    // Basic email validation before attempting to send
    if (submissionData.email && submissionData.email.includes('@')) {
      sendConfirmationEmail(submissionData);
      console.log(`Confirmation email successfully sent to ${submissionData.fullName} (${submissionData.email})`);
    } else {
      console.warn(`Invalid or missing email for ${submissionData.fullName}. Skipping email send.`);
    }

  } catch (error) {
    // Log the full error stack for better debugging
    console.error(`An error occurred in onFormSubmit: ${error.toString()}\nStack: ${error.stack}`);
  }
}

//================================================================
// EMAIL HELPER FUNCTION
//================================================================
/**
 * Composes and sends the confirmation email.
 *
 * @param {Object} data The extracted submission data.
 */
function sendConfirmationEmail(data) {
  const {
    fullName,
    email,
    transactionType,
    dateOfExpense,
    dateNeeded,
    category,
    description,
    amount,
    receiptLink,
    paymentWorks,
    otherComments
  } = data;

  const subject = `OChEGS Expense Request Received — ${transactionType} for $${amount}`;
  const firstName = fullName.split(' ')[0] || 'there';

  // Using template literals for a super clean email body
  const body = `
Hello ${firstName},

Thanks for submitting your expense request with the Organization of Chemical Engineering Graduate Students (OChEGS). Here are the details we received:

📝 Transaction Type: ${transactionType}
📅 Date of Purchase/Expense: ${dateOfExpense}
📆 When Do You Need the Funds?: ${dateNeeded}
💸 Amount Requested: $${amount}
📂 Expense Category: ${category}
💰 UH PaymentWorks Email for refund: ${paymentWorks}
📋 Description: ${description}
📎 Receipt: ${receiptLink}
${otherComments ? `🗒️ **Other Comments:** ${otherComments}` : ''}

Your request is now in the queue for review and approval. We'll reach out if we need any extra info or once it’s processed.

Got questions? Hit us up at uh.ochegs@gmail.com

Stay lit and keep crushing that ChemE grind! 🚀

— OChEGS Treasurer
`;

  // The trim() removes any leading/trailing whitespace from the template literal
  MailApp.sendEmail(email, subject, body.trim());
}
