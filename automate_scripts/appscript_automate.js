/**
 * This script sends both a confirmation email AND a calendar invite
 * to every person who submits the form.
 */

//================================================================
// MAIN CONTROLLER FUNCTION
//================================================================
function formResponse(e) {
  try {
    // Get all responses from the form using their question titles.
    const responses = e.namedValues;
    if (!responses) {
      console.log("This function is designed to run from a form submission trigger. Please test by submitting the live form.");
      return;
    }

    // Get the respondent's details.
    const firstName = responses['Your first name'][0] || '';
    const lastName = responses['Your last name'][0] || '';
    const email = responses['Your email'][0] || '';
    const fullName = `${firstName} ${lastName}`.trim();

    // Proceed only if a valid email was found.
    if (email && email.includes('@')) {
      // Call both helper functions to get the job done.
      sendConfirmationEmail(fullName, email);
      createCalendarInvite(email); // The calendar invite just needs the email.
      
      console.log(`Successfully processed email and calendar invite for ${fullName}.`);
    } else {
      console.log(`Skipped processing for ${fullName}. Reason: A valid email was not found.`);
    }
  } catch (error) {
    console.error(`Oof, something went wrong in the main function: ${error.toString()}`);
  }
}


//================================================================
// HELPER FUNCTION #1: SENDS THE EMAIL
//================================================================
function sendConfirmationEmail(name, email) {
  const eventTitle = "40th Annual ChBE Research Symposium";
  const subject = `Confirmed: Your RSVP for the ${eventTitle}`;
  const firstName = name.split(' ')[0];

  const body = `Dear ${firstName},\n\n` +
               `Thank you for your RSVP! This email confirms your attendance for the 40th Annual ChBE Research Symposium.\n\n` +
               `You will also receive a separate calendar invitation to automatically add this event to your calendar.\n\n` +
               `Here are the event details for your reference:\n\n` +
               `📅 **Date:** Friday, October 10th, 2025\n` +
               `🕐 **Time:** 8:00 AM – 5:00 PM CDT\n` +
               `📍 **Venue:** Houston Room, Student Center South\n` +
               `      4455 University Dr, Houston TX, 77204\n\n` +
               `To explore highlights from our previous symposiums, kindly visit our website: https://ochegs.chee.uh.edu/events/\n\n` +
               `We look forward to seeing you there!`;

  MailApp.sendEmail(email, subject, body);
}


//================================================================
// HELPER FUNCTION #2: SENDS THE CALENDAR INVITE
//================================================================
function createCalendarInvite(email) {
  const eventTitle = "40th Annual ChBE Research Symposium";
  const startTime = new Date('2025-10-10T08:00:00-05:00');
  const endTime = new Date('2025-10-10T17:00:00-05:00');
  const eventLocation = "Houston Room, Student Center South, 4455 University Dr, Houston TX, 77204";
  
  const description = "This is your official calendar invitation for the 40th Annual ChBE Research Symposium.\n\n" +
                      "For full details, please refer to your confirmation email or visit our website: https://ochegs.chee.uh.edu/";

  // This creates the event and invites the guest, sending them an invitation email.
  CalendarApp.createEvent(eventTitle, startTime, endTime, {
    description: description,
    location: eventLocation,
    guests: email,
    sendInvites: true
  });
}
