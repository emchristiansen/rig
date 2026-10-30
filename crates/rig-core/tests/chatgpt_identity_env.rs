//! A ChatGPT configuration read from the environment carries the caller's
//! exact identity when both of the dialect's identity variables give it,
//! none when neither does, and is refused, naming the missing variable,
//! when only one does.
//!
//! The variables are process-wide, which is why this test is its own binary.

#![allow(clippy::expect_used, clippy::panic)]

use rig_core::client::env::EnvError;
use rig_core::providers::chatgpt;
use rig_core::providers::openai::OpenAIConfig;
use rig_core::providers::openai::wire::CallerIdentity;

#[test]
fn a_chatgpt_configuration_from_the_environment_carries_the_caller_s_identity() {
    // SAFETY: this test binary has one test and no other threads read the
    // environment while it runs.
    unsafe {
        // Every other variable the dialect reads is unset, so the whole
        // configuration is the dialect's defaults plus what is set here.
        for name in [
            "CHATGPT_API_BASE",
            "OPENAI_CHATGPT_API_BASE",
            "CHATGPT_ACCOUNT_ID",
            "CHATGPT_DEFAULT_INSTRUCTIONS",
        ] {
            std::env::remove_var(name);
        }
        std::env::set_var("CHATGPT_ACCESS_TOKEN", "tok");
        std::env::set_var("CHATGPT_ORIGINATOR", "client_exec");
        std::env::set_var("CHATGPT_USER_AGENT", "client_exec/1.2.3");
    }
    assert_eq!(
        OpenAIConfig::from_env_with(&chatgpt::DIALECT).expect("both identity variables are set"),
        OpenAIConfig::with_key(&chatgpt::DIALECT, "tok").with_caller_identity(
            CallerIdentity::new("client_exec", "client_exec/1.2.3", None)
                .expect("a valid identity")
        ),
    );

    // SAFETY: as above.
    unsafe {
        std::env::remove_var("CHATGPT_USER_AGENT");
    }
    let Err(EnvError::Invalid { name, detail }) = OpenAIConfig::from_env_with(&chatgpt::DIALECT)
    else {
        panic!("an originator without a user agent is refused");
    };
    assert_eq!(name, "CHATGPT_USER_AGENT");
    assert_eq!(
        detail,
        "`CHATGPT_ORIGINATOR` is set, so `CHATGPT_USER_AGENT` must be set too: this gateway \
         takes the caller's exact identity, never half of one"
    );

    // SAFETY: as above.
    unsafe {
        std::env::remove_var("CHATGPT_ORIGINATOR");
    }
    assert_eq!(
        OpenAIConfig::from_env_with(&chatgpt::DIALECT).expect("no identity variable is set"),
        OpenAIConfig::with_key(&chatgpt::DIALECT, "tok"),
    );
}
