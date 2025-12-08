"""
Email utilities for FactuAI Backend

Handles sending emails via Resend API.
"""

import os
import resend
from utils.logging import get_logger

logger = get_logger(__name__)

# Initialize Resend API key from environment
resend.api_key = os.environ.get("RESEND_API_KEY")


def send_password_reset_email(email: str, token: str):
    """
    Send password reset email using Resend SDK
    
    Args:
        email: Recipient email address
        token: Password reset token
        
    Returns:
        True if email sent successfully, False otherwise
    """
    try:
        reset_link = f"{os.environ.get('FRONTEND_URL', 'http://localhost:3000')}/reset-password?token={token}"
        from_email = os.environ.get('FROM_EMAIL', 'noreply@resend.dev')

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Reset your FactuAI password</title>
        </head>
        <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 10px; margin-bottom: 30px;">
                <h1 style="color: white; margin: 0; font-size: 28px; font-weight: bold;">
                    🔍 FactuAI
                </h1>
                <p style="color: #f0f0f0; margin: 10px 0 0 0; font-size: 16px;">
                    Password Reset Request
                </p>
            </div>
            
            <div style="background: #f8f9fa; padding: 30px; border-radius: 10px; border-left: 4px solid #667eea;">
                <h2 style="color: #333; margin-top: 0;">Reset Your Password</h2>
                <p style="font-size: 16px; margin-bottom: 25px;">
                    You requested a password reset for your FactuAI account. Click the button below to create a new password.
                </p>
                
                <div style="text-align: center; margin: 30px 0;">
                    <a href="{reset_link}" 
                       style="display: inline-block; padding: 15px 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-decoration: none; border-radius: 8px; font-weight: bold; font-size: 16px; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);">
                        Reset Password
                    </a>
                </div>
                
                <p style="font-size: 14px; color: #666; margin-top: 25px;">
                    <strong>Security Note:</strong> This link will expire in 30 minutes for your security.
                </p>
                
                <hr style="border: none; border-top: 1px solid #e9ecef; margin: 25px 0;">
                
                <p style="font-size: 14px; color: #666;">
                    If you didn't request this password reset, you can safely ignore this email. Your password will not be changed.
                </p>
                
                <p style="font-size: 12px; color: #999; margin-top: 20px;">
                    If the button doesn't work, copy and paste this link into your browser:<br>
                    <span style="word-break: break-all;">{reset_link}</span>
                </p>
            </div>
            
            <div style="text-align: center; margin-top: 30px; padding: 20px; color: #666; font-size: 12px;">
                <p>© 2025 FactuAI. All rights reserved.</p>
                <p>This email was sent to {email}</p>
            </div>
        </body>
        </html>
        """

        # Use the correct Resend Python SDK syntax from official docs
        params = {
            "from": f"FactuAI <{from_email}>",
            "to": [email],
            "subject": "Reset your FactuAI password",
            "html": html_content
        }

        response = resend.Emails.send(params)
        
        logger.info(f"Password reset email sent successfully to {email}")
        logger.info(f"Email response: {response}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send password reset email to {email}: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        if hasattr(e, 'response'):
            logger.error(f"Response details: {e.response}")
        return False
